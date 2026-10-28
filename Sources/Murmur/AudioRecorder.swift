import AVFoundation
import AudioToolbox
import CoreAudio
import CoreGraphics
import Foundation

/// Captures microphone audio into a temporary file while the hotkey is held.
///
/// Deliberately simple: the engine starts on key-down and stops on release.
/// Two "improvements" were tried and reverted after breaking things:
/// - setVoiceProcessingEnabled: its echo canceller ducks/mutes other apps'
///   audio system-wide and can feed the recognizer silence.
/// - A warm always-on engine with a pre-roll ring buffer: wedged the engine
///   so recording never started.
///
/// Which microphone, and what happens when the hardware changes mid-recording
/// or the microphone delivers nothing, is decided per recording and needs no
/// setting — see `MicrophoneChoice`, `onInterrupted` and `deliveredOnlyZeros`.
final class AudioRecorder {
    /// Replaced by a fresh one when the microphone changes to or from a Bluetooth headset: an engine keeps the sample
    /// rate it was first set up with (48 kHz for the Mac's microphone), a headset in call mode delivers 24 kHz, and
    /// macOS then refuses to start ("formats don't match", -10868) — which is what "Couldn't start the microphone" with
    /// AirPods was.
    private var engine = AVAudioEngine()
    /// The input device the current engine was last pointed at.
    private var engineDeviceID: AudioDeviceID?
    private var file: AVAudioFile?
    private(set) var currentFileURL: URL?
    private(set) var isRecording = false
    /// Audio frames actually captured. A recording that ends on zero means
    /// the microphone delivered nothing, which is worth reporting rather
    /// than passing an empty file to the recognizer and shrugging.
    private(set) var capturedFrames: AVAudioFramePosition = 0
    /// The format the tap really ran at, for diagnostics.
    private(set) var capturedFormat: AVAudioFormat?
    /// True once *enough total time* above the silence floor has been
    /// seen. A recording that never clears it isn't "no audio" (frames
    /// were captured — room tone, mic self-noise) but there's no actual
    /// speech in it either. Whisper models hallucinate on exactly this
    /// input — trained partly on YouTube transcripts, they'll fill true
    /// silence with "Thank you." or similar sign-off phrases rather than
    /// admit there's nothing there. Catching it here means every engine
    /// benefits, not just whichever one happens to hallucinate.
    ///
    /// This used to latch true the moment *any single* buffer crossed the
    /// floor — "was there ever a moment above threshold" — and a real
    /// incident showed why that's too weak: one click, pop, or breath is
    /// loud enough on its own to latch it permanently, even across an
    /// otherwise-silent 1.5 second hold, and the recognizer still
    /// hallucinated "Thank you." from what was actually silence.
    /// Requiring a minimum cumulative *duration* above the floor — not
    /// just an instant — is what the threshold below was actually tuned
    /// against ("comfortably below even quiet speech"): a transient is
    /// over almost immediately, a spoken word isn't.
    private(set) var hasSignal = false
    /// Frames whose own buffer cleared the silence floor, summed across
    /// the whole recording (not just whether one ever did).
    private var signalFrameCount: AVAudioFramePosition = 0
    /// -40dBFS. Comfortably below even quiet speech, comfortably above a
    /// silent room's noise floor — picked to avoid false positives on real
    /// (if soft) speech, not tuned to catch every last whisper.
    private let silenceRMSThreshold: Float = 0.01
    /// Below this much cumulative above-threshold audio, the recording
    /// counts as silence even if isolated buffers spiked. At the tap's
    /// 4096-frame buffer size (~85ms at 48kHz), this needs at least two
    /// consecutive above-floor buffers — long enough that a single click
    /// or mouth-noise can't clear it alone, short enough that even a
    /// one-syllable word comfortably does.
    private let minimumSignalDuration: Double = 0.15

    /// Optional live feed of every captured buffer, for callers that need to
    /// watch an in-progress recording (hands-free auto-stop) rather than
    /// just the completed file. `nil` by default — push-to-talk and
    /// hands-free sessions with no listener pay nothing extra. Called
    /// synchronously from the real-time tap callback below, so whatever's
    /// attached here must return immediately, same as the rest of that
    /// callback — no async work, no blocking.
    var onLiveBuffer: ((AVAudioPCMBuffer) -> Void)?

    /// A second, independent live feed, for the HUD's live-transcript
    /// preview — separate from `onLiveBuffer` because the two run on
    /// different schedules (hands-free-with-auto-stop sessions only, vs.
    /// every recording) and must never clobber each other by sharing one
    /// slot the way a single property would. Same real-time contract as
    /// `onLiveBuffer`: called synchronously from the tap callback, so
    /// whatever's attached here must return immediately.
    var onLivePreviewBuffer: ((AVAudioPCMBuffer) -> Void)?
    /// A third, for transcribing a long dictation while it's still being spoken (`LiveChunkSession`): whisper.cpp
    /// dictations only. Same real-time contract — it copies the samples and returns.
    var onLiveChunkBuffer: ((AVAudioPCMBuffer) -> Void)?
    /// How loud each buffer is, 0…1 (`level(of:)`) — what the HUD's bars show while you talk. Same real-time contract as
    /// `onLiveBuffer`: called from the tap callback, so it must hand the value off and return.
    var onLevel: ((Float) -> Void)?

    /// Called on the main thread when the audio hardware changes under a recording — a headset connecting, a dock,
    /// the Mac waking. The engine stops itself when that happens (Apple: it "stops, uninitializes itself"), so the
    /// rest of the dictation used to vanish without a word; the caller finishes it with what was captured instead.
    var onInterrupted: (() -> Void)?
    private var configurationObserver: NSObjectProtocol?
    private var startedAt: Date?
    /// Switching the input device at the start of a recording can post the same notification — that one is ours.
    private var deviceSwitchedAt: Date?
    /// Frames that weren't pure digital zero. A microphone that is denied, cut off in hardware (a closed MacBook
    /// lid) or held by another device delivers exact zeros rather than an error, so "a quiet room" and "the
    /// microphone isn't delivering" can be told apart.
    private var nonZeroFrameCount: AVAudioFramePosition = 0
    private var firstBufferAt: Date?
    private var firstSoundAt: Date?
    /// How the input this recording used is connected (`kAudioDeviceTransportType…`), for the message when it
    /// delivered nothing.
    private(set) var inputTransport: UInt32 = 0
    /// Run by the next tap buffer — `stopAfterTail`'s signal that the audio in flight at the key's release is in.
    private let tailLock = NSLock()
    private var tailWaiter: (() -> Void)?

    /// This recording's start-up, for the command line's check: milliseconds from start to the first buffer and to the
    /// first buffer with sound in it.
    private(set) var lastStartup: (firstBuffer: Int?, firstSound: Int?) = (nil, nil)
    /// Keep each recording's start-up in `MicTimingLog` — the dictation recorder only.
    var recordsStartup = false

    /// Settings ▸ Dictation ▸ "Use your AirPods’ microphone" (`useBluetoothMicrophone`): record from a connected
    /// Bluetooth headset instead of the Mac's own microphone. Off by default — see `MicrophoneChoice`.
    static var prefersBluetooth: Bool { UserDefaults.standard.bool(forKey: "useBluetoothMicrophone") }
    /// For the command line's comparison: this recorder always uses this kind of microphone.
    enum ForcedInput { case builtIn, bluetooth }
    var forcedInput: ForcedInput?
    /// Called once per recording, on the main thread, when the first buffer with real (non-zero) audio arrives —
    /// for a Bluetooth headset that is the moment it has finished switching into call mode and can hear you.
    var onFirstAudio: (() -> Void)?
    /// A Bluetooth headset stays in call mode this long after a dictation, so the next one starts at once instead of
    /// waiting seconds for the switch again (VoiceOS keeps its "warm stream" the same way).
    static let bluetoothWarmSeconds: TimeInterval = 30
    private var warmStopWork: DispatchWorkItem?
    /// Turns a buffer that arrives in a new format (a headset changing sample rate as it switches into call mode)
    /// into the format the file was created with.
    private var converter: AVAudioConverter?
    /// How often the engine was restarted during this recording after a Bluetooth headset changed its format.
    private(set) var restartsThisRecording = 0

    /// Real (non-zero) audio has arrived in this recording.
    var hasDeliveredAudio: Bool { nonZeroFrameCount > 0 }
    /// Frames arrived and every one was exact zero: the microphone isn't delivering, it isn't a quiet room.
    var deliveredOnlyZeros: Bool { capturedFrames > 0 && nonZeroFrameCount == 0 }
    var inputIsBluetooth: Bool {
        inputTransport == kAudioDeviceTransportTypeBluetooth || inputTransport == kAudioDeviceTransportTypeBluetoothLE
    }

    init() {
        observeConfigurationChanges()
    }

    private func observeConfigurationChanges() {
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in self?.configurationChanged() }
    }

    /// A fresh engine, for a different microphone.
    private func replaceEngine() {
        warmStopWork?.cancel()
        warmStopWork = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engine = AVAudioEngine()
        engineDeviceID = nil
        observeConfigurationChanges()
    }

    deinit {
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
    }

    private func configurationChanged() {
        guard isRecording else { return }
        if let switched = deviceSwitchedAt, Date().timeIntervalSince(switched) < 1 { return }
        // A Bluetooth headset that has just been asked for its microphone switches into call mode and changes its
        // format seconds later — that is the start of this recording, not the hardware going away. Start listening
        // again on the same headset rather than ending the dictation (which is why AirPods used to "not work at
        // first").
        if inputIsBluetooth, restartsThisRecording < 3, let startedAt, Date().timeIntervalSince(startedAt) < 12 {
            restartsThisRecording += 1
            do {
                let input = engine.inputNode
                input.removeTap(onBus: 0)
                installTap(on: input)
                engine.prepare()
                try engine.start()
                dictationLog.notice("mic: the Bluetooth headset switched into call mode; listening again")
                return
            } catch {
                dictationLog.error("mic: couldn't listen again after the headset switched: \(error.localizedDescription, privacy: .public)")
            }
        }
        dictationLog.error("mic: the audio hardware changed during a recording; finishing with what was captured")
        onInterrupted?()
    }

    static func requestMicrophoneAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        default:
            return false
        }
    }

    func start() throws {
        guard !isRecording else { return }
        warmStopWork?.cancel()
        warmStopWork = nil
        restartsThisRecording = 0
        converter = nil

        // Which microphone: the one chosen in System Settings, unless it's a
        // Bluetooth headset or a virtual device — see `MicrophoneChoice`. It may replace the engine.
        chooseMicrophone()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw NSError(
                domain: "Murmur", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No microphone input available"])
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("murmur-\(UUID().uuidString).caf")

        // Installing a tap on a bus that already has one raises an
        // NSException from AVFoundation — which Swift cannot catch, so it
        // aborts the whole process. `removeTap` on a bus with no tap is a
        // no-op, making this the cheap way to guarantee the bus is clear.
        input.removeTap(onBus: 0)

        file = nil
        currentFileURL = url
        capturedFrames = 0
        capturedFormat = nil
        hasSignal = false
        signalFrameCount = 0
        nonZeroFrameCount = 0
        firstBufferAt = nil
        firstSoundAt = nil
        startedAt = Date()
        tailLock.lock()
        tailWaiter = nil
        tailLock.unlock()

        // `format: nil` means "whatever this bus is actually running at".
        //
        // Passing an explicit format is the other way installTap raises —
        // and a Bluetooth headset guarantees it eventually will. AirPods sit
        // at 48kHz for playback but drop to 24kHz the instant the mic is
        // engaged, so a format read a moment earlier is stale by the time
        // the tap installs: exception, process aborted.
        installTap(on: input)

        do {
            // A Bluetooth headset kept in call mode since the last dictation: the engine is still running.
            if !engine.isRunning {
                engine.prepare()
                try engine.start()
            }
        } catch {
            // Unwind completely. Leaving the tap installed while
            // `isRecording` stayed false was a latent landmine: the next
            // start() would sail past the guard above, install a second tap,
            // and kill the app — so one transient audio failure permanently
            // broke dictation until relaunch.
            input.removeTap(onBus: 0)
            engine.stop()
            file = nil
            currentFileURL = nil
            try? FileManager.default.removeItem(at: url)
            throw error
        }
        isRecording = true
    }

    /// The recording tap — installed at the start, and again when a Bluetooth headset restarts the engine.
    private func installTap(on input: AVAudioInputNode) {
        let url = currentFileURL ?? FileManager.default.temporaryDirectory.appendingPathComponent("murmur-\(UUID().uuidString).caf")
        // A Bluetooth headset in call mode: tap at the rate it really delivers (24 kHz for AirPods) — the engine's own
        // idea of the format can still be the 48 kHz of music mode, and starting with that fails (-10868).
        // The same after `chooseMicrophone` moves the node off macOS's default in/out aggregate (a 48 kHz USB
        // interface while a 44.1 kHz Bluetooth speaker is the output): the node keeps the aggregate's 44.1 kHz.
        let hardware = input.inputFormat(forBus: 0)
        let node = input.outputFormat(forBus: 0)
        let stale = node.sampleRate != hardware.sampleRate || node.channelCount != hardware.channelCount
        let tapFormat: AVAudioFormat? = (inputIsBluetooth || stale) && hardware.sampleRate > 0 && hardware.channelCount > 0
            ? hardware : nil
        input.installTap(onBus: 0, bufferSize: 4096, format: tapFormat) { [weak self] incoming, _ in
            guard let self else { return }
            // The same format as the file, whatever the headset is sending now.
            guard let buffer = self.matchingFileFormat(incoming) else { return }
            // The file is created from the FIRST buffer's real format rather
            // than the format read above, for the same reason. When those
            // disagreed, every write threw, `try?` swallowed it, and the
            // recording ended up as a header with no audio — a silent
            // failure that looked exactly like the app being broken.
            if self.file == nil {
                self.file = try? AVAudioFile(
                    forWriting: url, settings: buffer.format.settings)
                self.capturedFormat = buffer.format
            }
            try? self.file?.write(from: buffer)
            self.capturedFrames += AVAudioFramePosition(buffer.frameLength)
            if self.firstBufferAt == nil { self.firstBufferAt = Date() }
            if Self.containsNonZero(buffer) {
                if self.nonZeroFrameCount == 0, let onFirstAudio = self.onFirstAudio {
                    DispatchQueue.main.async(execute: onFirstAudio)
                }
                self.nonZeroFrameCount += AVAudioFramePosition(buffer.frameLength)
            }
            if Self.hasSignal(in: buffer, above: self.silenceRMSThreshold) {
                self.signalFrameCount += AVAudioFramePosition(buffer.frameLength)
                if self.firstSoundAt == nil { self.firstSoundAt = Date() }
            }
            self.onLiveBuffer?(buffer)
            self.onLivePreviewBuffer?(buffer)
            self.onLiveChunkBuffer?(buffer)
            if let onLevel = self.onLevel { onLevel(Self.level(of: buffer)) }
            self.tailLock.lock()
            let tailWaiter = self.tailWaiter
            self.tailWaiter = nil
            self.tailLock.unlock()
            tailWaiter?()
        }

    }

    /// `incoming` as it can be written to the file: unchanged while the format is the file's, converted once the
    /// device has changed format mid-recording, nil when it can't be converted.
    private func matchingFileFormat(_ incoming: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let target = capturedFormat, incoming.format != target else { return incoming }
        if converter == nil || converter?.inputFormat != incoming.format {
            converter = AVAudioConverter(from: incoming.format, to: target)
        }
        guard let converter else { return nil }
        let capacity = AVAudioFrameCount(Double(incoming.frameLength) * target.sampleRate / incoming.format.sampleRate) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return nil }
        var handed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if handed { status.pointee = .noDataNow; return nil }
            handed = true
            status.pointee = .haveData
            return incoming
        }
        return error == nil && output.frameLength > 0 ? output : nil
    }

    /// Stops recording and returns the captured audio file URL,
    /// or nil if nothing was recorded.
    @discardableResult
    func stop() -> URL? {
        guard isRecording else { return nil }
        engine.inputNode.removeTap(onBus: 0)
        stopEngineOrKeepWarm()
        isRecording = false
        file = nil
        tailLock.lock()
        tailWaiter = nil
        tailLock.unlock()
        // How long the microphone took to deliver, by connection — durations only. What decides whether a warmer
        // microphone is ever worth its always-on indicator (lost first words are 0.4% of errors today).
        if let startedAt {
            func milliseconds(_ date: Date?) -> Int? { date.map { Int($0.timeIntervalSince(startedAt) * 1000) } }
            func ms(_ date: Date?) -> String { milliseconds(date).map(String.init) ?? "-" }
            lastStartup = (milliseconds(firstBufferAt), milliseconds(firstSoundAt))
            if recordsStartup {
                MicTimingLog.record(transport: Self.transportName(inputTransport), firstBuffer: lastStartup.firstBuffer,
                                    firstSound: lastStartup.firstSound, onlyZeros: deliveredOnlyZeros)
            }
            // Notice, not info: macOS throws info-level lines away, and this one is what decides the warm-mic question.
            dictationLog.notice(
                "mic: first buffer \(ms(self.firstBufferAt), privacy: .public) ms, first sound \(ms(self.firstSoundAt), privacy: .public) ms after key-down; transport \(Self.transportName(self.inputTransport), privacy: .public)\(self.deliveredOnlyZeros ? ", EXACT ZEROS" : "", privacy: .public)")
        }
        startedAt = nil
        hasSignal = Self.hasSustainedSignal(
            signalFrames: signalFrameCount,
            sampleRate: capturedFormat?.sampleRate ?? 0,
            minimumDuration: minimumSignalDuration)
        let url = currentFileURL
        currentFileURL = nil
        return url
    }

    /// A Bluetooth headset chosen on purpose stays in call mode for `bluetoothWarmSeconds` (the engine keeps running
    /// with no tap, so nothing is recorded); anything else stops at once.
    private func stopEngineOrKeepWarm() {
        warmStopWork?.cancel()
        guard inputIsBluetooth, Self.prefersBluetooth || forcedInput == .bluetooth else {
            engine.stop()
            return
        }
        let work = DispatchWorkItem { [weak self] in
            guard let self, !self.isRecording else { return }
            self.engine.stop()
            dictationLog.info("mic: let the Bluetooth headset go back to music mode")
        }
        warmStopWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.bluetoothWarmSeconds, execute: work)
    }

    /// Stops once the audio still in flight when the key came up has arrived — the tap hands audio over in blocks of
    /// about a tenth of a second, and stopping at once threw away the part of the block holding the end of the last
    /// word (Handy fixed the same loss in its 0.9.7, PR #1958). Waits for the next block, at most `maxWait`, then
    /// stops; `completion` runs on the main thread with what `stop()` returns.
    func stopAfterTail(maxWait: TimeInterval = 0.12, completion: @escaping (URL?) -> Void) {
        guard isRecording else { completion(nil); return }
        var finished = false
        let finish: () -> Void = { [weak self] in
            guard !finished else { return }
            finished = true
            completion(self?.stop())
        }
        tailLock.lock()
        tailWaiter = { DispatchQueue.main.async(execute: finish) }
        tailLock.unlock()
        DispatchQueue.main.asyncAfter(deadline: .now() + maxWait, execute: finish)
    }

    /// Stops and deletes the in-progress recording.
    func cancel() {
        if let url = stop() {
            try? FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Silence detection

    /// Any sample that isn't exactly zero. Real microphones always carry some noise; a stream of perfect zeros means
    /// the device isn't delivering. An unrecognised format counts as delivering (fails open).
    static func containsNonZero(_ buffer: AVAudioPCMBuffer) -> Bool {
        guard let samples = buffer.floatChannelData?[0] else { return true }
        for index in 0..<Int(buffer.frameLength) where samples[index] != 0 { return true }
        return false
    }

    /// RMS energy of one buffer against a threshold. The tap's format comes
    /// from whatever the live device reports (see `start()`), which in
    /// practice is always deinterleaved Float32 for an `AVAudioInputNode` —
    /// but "in practice" isn't "guaranteed", so an unrecognised format fails
    /// open (`true`) rather than silently treating real speech as silence.
    static func hasSignal(in buffer: AVAudioPCMBuffer, above threshold: Float) -> Bool {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return false }
        guard let channelData = buffer.floatChannelData else { return true }

        var sumOfSquares: Float = 0
        let samples = channelData[0]
        for i in 0..<frameCount {
            let sample = samples[i]
            sumOfSquares += sample * sample
        }
        let rms = (sumOfSquares / Float(frameCount)).squareRoot()
        return rms > threshold
    }

    /// How loud one buffer is, for the HUD's bars: its RMS in dBFS mapped onto 0…1 — -50 dB (room tone, where the bars
    /// rest as dots) to -15 dB (loud speech, full height). So a flat line honestly means the microphone isn't hearing
    /// you. Same format caveat as `hasSignal`: an unrecognised format reads as silent rather than guessing.
    static func level(of buffer: AVAudioPCMBuffer) -> Float {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0, let channelData = buffer.floatChannelData else { return 0 }
        var sumOfSquares: Float = 0
        let samples = channelData[0]
        for i in 0..<frameCount {
            let sample = samples[i]
            sumOfSquares += sample * sample
        }
        let rms = (sumOfSquares / Float(frameCount)).squareRoot()
        let decibels = 20 * log10(max(rms, 1e-7))
        return min(max((decibels + 50) / 35, 0), 1)
    }

    /// Whether accumulated above-floor audio time (from summing
    /// `hasSignal(in:above:)` across every buffer in a recording) clears
    /// the minimum a real spoken word needs — as opposed to a single
    /// buffer's transient spike. `sampleRate <= 0` (no audio ever
    /// captured) can't clear any duration, so it's silence, not a crash.
    static func hasSustainedSignal(
        signalFrames: AVAudioFramePosition, sampleRate: Double, minimumDuration: Double
    ) -> Bool {
        guard sampleRate > 0 else { return false }
        return Double(signalFrames) / sampleRate >= minimumDuration
    }

    // MARK: - Device selection

    /// Points the input node at the microphone `MicrophoneChoice` picks. Best-effort: any failure leaves the node on
    /// whatever it was already using.
    private func chooseMicrophone() {
        // The device this recording should use, decided before touching the engine: going to or from a Bluetooth
        // headset needs a fresh engine (see `engine`).
        let wanted = Self.wantedInputDevice(forcedInput: forcedInput)
        // Also when the engine is still holding a headset in call mode (`stopEngineOrKeepWarm`) and this recording
        // wants another microphone: measured 2026-10-14, the Mac's microphone delivered nothing at all while the same
        // process held AirPods in call mode.
        if let wanted, engineDeviceID != wanted.id, MicrophoneChoice.isBluetooth(wanted) || inputIsBluetooth {
            replaceEngine()
            dictationLog.info("mic route: a fresh audio engine for the \(Self.transportName(wanted.transport), privacy: .public) microphone")
        }
        let input = engine.inputNode
        guard let unit = input.audioUnit else {
            dictationLog.error("mic route: no input AudioUnit yet")
            return
        }
        let devices = Self.inputDevices()
        let defaultID = Self.defaultInputDeviceID()
        var choice = MicrophoneChoice.choose(
            systemDefault: devices.first { $0.id == defaultID }, all: devices,
            builtInDisplayActive: Self.builtInDisplayActive(), prefersBluetooth: Self.prefersBluetooth)
        switch forcedInput {
        case .builtIn: choice = devices.first { $0.transport == kAudioDeviceTransportTypeBuiltIn }?.id ?? choice
        case .bluetooth: choice = devices.first { MicrophoneChoice.isBluetooth($0) }?.id ?? choice
        case nil: break
        }
        // Debug only (the command line's microphone check): "builtin" to time the Mac's own microphone.
        if ProcessInfo.processInfo.environment["MURMUR_MIC"] == "builtin" {
            choice = devices.first { $0.transport == kAudioDeviceTransportTypeBuiltIn }?.id ?? choice
        }
        guard var deviceID = choice ?? defaultID else {
            dictationLog.info("mic route: no input device found, leaving the engine as it is")
            return
        }
        inputTransport = devices.first { $0.id == deviceID }?.transport ?? 0
        var current = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let read = AudioUnitGetProperty(
            unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &current, &size)
        guard read != noErr || current != deviceID else {
            engineDeviceID = deviceID
            return
        }
        deviceSwitchedAt = Date()
        let status = AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0,
            &deviceID, UInt32(MemoryLayout<AudioDeviceID>.size))
        engineDeviceID = deviceID
        dictationLog.info(
            "mic route: \(choice == nil ? "system default" : "instead of the default", privacy: .public) device \(deviceID) (\(Self.transportName(self.inputTransport), privacy: .public)) -> status \(status)")
    }

    /// The input a recording would use now, as a device — `MicrophoneChoice` plus the command line's forced input.
    private static func wantedInputDevice(forcedInput: ForcedInput?) -> MicrophoneChoice.Device? {
        let devices = inputDevices()
        let defaultID = defaultInputDeviceID()
        var choice = MicrophoneChoice.choose(
            systemDefault: devices.first { $0.id == defaultID }, all: devices,
            builtInDisplayActive: builtInDisplayActive(), prefersBluetooth: prefersBluetooth)
        switch forcedInput {
        case .builtIn: choice = devices.first { $0.transport == kAudioDeviceTransportTypeBuiltIn }?.id ?? choice
        case .bluetooth: choice = devices.first { MicrophoneChoice.isBluetooth($0) }?.id ?? choice
        case nil: break
        }
        let id = choice ?? defaultID
        return devices.first { $0.id == id }
    }

    /// Every device with an input side, with how it's connected.
    private static func inputDevices() -> [MicrophoneChoice.Device] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize) == noErr,
            dataSize > 0
        else { return [] }
        var deviceIDs = [AudioDeviceID](repeating: 0, count: Int(dataSize) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil,
            &dataSize, &deviceIDs) == noErr
        else { return [] }
        return deviceIDs.filter(hasInputStreams).map { .init(id: $0, transport: transportType($0)) }
    }

    private static func defaultInputDeviceID() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID) == noErr,
            deviceID != 0
        else { return nil }
        return deviceID
    }

    /// The Mac's chosen input in System Settings is a Bluetooth device — for Settings to say why Murmur doesn't simply
    /// use it (`MicrophoneChoice`).
    static func systemDefaultInputIsBluetooth() -> Bool {
        guard let id = defaultInputDeviceID() else { return false }
        let transport = transportType(id)
        return transport == kAudioDeviceTransportTypeBluetooth || transport == kAudioDeviceTransportTypeBluetoothLE
    }

    private static func transportType(_ deviceID: AudioDeviceID) -> UInt32 {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var transportType: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(
            deviceID, &address, 0, nil, &size, &transportType) == noErr
        else { return 0 }
        return transportType
    }

    /// The MacBook's own screen is on. With the lid closed the built-in microphone is disconnected in hardware
    /// (Apple Platform Security) but still listed, and records pure silence.
    private static func builtInDisplayActive() -> Bool {
        var count: UInt32 = 0
        guard CGGetActiveDisplayList(0, nil, &count) == .success, count > 0 else { return false }
        var displays = [CGDirectDisplayID](repeating: 0, count: Int(count))
        guard CGGetActiveDisplayList(count, &displays, &count) == .success else { return false }
        return displays.contains { CGDisplayIsBuiltin($0) != 0 }
    }

    static func transportName(_ transport: UInt32) -> String {
        switch transport {
        case kAudioDeviceTransportTypeBuiltIn: return "built-in"
        case kAudioDeviceTransportTypeUSB: return "USB"
        case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE: return "Bluetooth"
        case kAudioDeviceTransportTypeVirtual: return "virtual"
        case kAudioDeviceTransportTypeAggregate: return "aggregate"
        case kAudioDeviceTransportTypeDisplayPort, kAudioDeviceTransportTypeHDMI: return "display"
        case kAudioDeviceTransportTypeThunderbolt: return "Thunderbolt"
        case 0: return "unknown"
        default: return "other"
        }
    }

    /// A device can be built-in and output-only (the Mac's speakers) — this
    /// confirms it actually has an input side before we route to it.
    private static func hasInputStreams(_ deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioObjectPropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size) == noErr
        else { return false }
        return size > 0
    }

    // MARK: - Self test

    static func runSelfTest() -> Bool {
        var passed = true
        func check(_ ok: Bool, _ label: String) {
            if !ok { passed = false }
            print("\(ok ? "PASS" : "FAIL"): \(label)")
        }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000,
            channels: 1, interleaved: false)
        else {
            check(false, "could not construct a test audio format")
            return false
        }

        func buffer(_ samples: [Float]) -> AVAudioPCMBuffer {
            let buf = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
            buf.frameLength = AVAudioFrameCount(samples.count)
            let channel = buf.floatChannelData![0]
            for (i, sample) in samples.enumerated() { channel[i] = sample }
            return buf
        }

        let threshold: Float = 0.01

        let silence = buffer([Float](repeating: 0, count: 1000))
        check(!hasSignal(in: silence, above: threshold), "pure silence has no signal")

        // A real mic never reports exact zero — self-noise sits a bit above
        // it. This is what "pressed the key, said nothing" actually looks
        // like, not literal digital silence.
        let micNoiseFloor = buffer((0..<1000).map { _ in Float.random(in: -0.002...0.002) })
        check(!hasSignal(in: micNoiseFloor, above: threshold),
              "mic self-noise floor stays below the threshold")

        let speechLike = buffer((0..<1000).map { i in
            Float(sin(Double(i) * 0.1)) * 0.3
        })
        check(hasSignal(in: speechLike, above: threshold),
              "speech-amplitude signal is detected")

        let empty = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 0)!
        check(!hasSignal(in: empty, above: threshold), "an empty buffer has no signal")

        // Right at the threshold: a constant-amplitude signal's RMS equals
        // that amplitude, so this is the exact boundary the `>` comparison
        // has to get right in both directions.
        let justBelow = buffer([Float](repeating: threshold - 0.0001, count: 100))
        check(!hasSignal(in: justBelow, above: threshold), "just below threshold is silence")
        let justAbove = buffer([Float](repeating: threshold + 0.0001, count: 100))
        check(hasSignal(in: justAbove, above: threshold), "just above threshold is signal")

        // Reproduces the actual incident: a real recording held for 1.5s at
        // 48kHz, where only one 4096-frame tap buffer (~85ms) ever crossed
        // the RMS floor — a click or breath, not speech. The old "was there
        // ever a moment above threshold" latch called this signal; it isn't.
        let minDuration = 0.15
        let oneClickBuffer: AVAudioFramePosition = 4096
        check(!hasSustainedSignal(
            signalFrames: oneClickBuffer, sampleRate: 48000, minimumDuration: minDuration),
              "one 85ms buffer above the floor isn't sustained signal")

        // Two-plus consecutive buffers — what an actual spoken word looks
        // like — clears it.
        let wordLength: AVAudioFramePosition = 8192
        check(hasSustainedSignal(
            signalFrames: wordLength, sampleRate: 48000, minimumDuration: minDuration),
              "170ms of above-floor audio is sustained signal")

        check(!hasSustainedSignal(signalFrames: 0, sampleRate: 48000, minimumDuration: minDuration),
              "zero signal frames is silence")
        check(!hasSustainedSignal(signalFrames: 999_999, sampleRate: 0, minimumDuration: minDuration),
              "an unknown sample rate can't clear any duration")

        // Exact boundary: 0.15s at 48kHz is 7200 frames.
        check(!hasSustainedSignal(signalFrames: 7199, sampleRate: 48000, minimumDuration: minDuration),
              "just under the duration floor is silence")
        check(hasSustainedSignal(signalFrames: 7200, sampleRate: 48000, minimumDuration: minDuration),
              "exactly the duration floor is signal")

        return passed
    }
}

/// Which microphone a dictation records from. Automatic, nothing to set:
///  1. with the lid open, the Mac's own microphone — it starts fastest, and whatever is said while a microphone is
///     still starting is lost. Measured 2026-09-30: the built-in mic delivered its first audio 155 ms after the key,
///     a Studio Display microphone (USB) 555 ms — for the day Murmur followed the system default to the Studio
///     Display, every dictation lost about 0.4 s more at the start;
///  2. otherwise (lid closed — the built-in mic is then disconnected in hardware but still listed, and records pure
///     silence) the microphone chosen in System Settings when it's wired, else the first wired one. Never a Bluetooth
///     headset while a wired one exists (switching it into call mode ducks what's playing and can outlast a quick
///     press) or a virtual device (often silent unless its app is running);
///  3. with nothing wired, the system default after all.
/// `nil` means "the system default".
enum MicrophoneChoice {
    struct Device: Equatable {
        let id: AudioDeviceID
        let transport: UInt32
    }

    static let wired: Set<UInt32> = [
        kAudioDeviceTransportTypeBuiltIn, kAudioDeviceTransportTypeUSB, kAudioDeviceTransportTypeDisplayPort,
        kAudioDeviceTransportTypeHDMI, kAudioDeviceTransportTypeThunderbolt, kAudioDeviceTransportTypePCI,
        kAudioDeviceTransportTypeFireWire, kAudioDeviceTransportTypeAVB,
    ]

    static func isBluetooth(_ device: Device) -> Bool {
        device.transport == kAudioDeviceTransportTypeBluetooth || device.transport == kAudioDeviceTransportTypeBluetoothLE
    }

    /// `prefersBluetooth` (the user's "Use your AirPods’ microphone"): a connected Bluetooth headset wins — the one
    /// chosen in System Settings if that is one, else the first connected.
    static func choose(systemDefault: Device?, all: [Device], builtInDisplayActive: Bool,
                       prefersBluetooth: Bool = false) -> AudioDeviceID? {
        if prefersBluetooth {
            if let systemDefault, isBluetooth(systemDefault) { return nil }
            if let headset = all.first(where: isBluetooth) { return headset.id }
        }
        func usable(_ device: Device) -> Bool {
            wired.contains(device.transport)
                && (device.transport != kAudioDeviceTransportTypeBuiltIn || builtInDisplayActive)
        }
        if builtInDisplayActive, let builtIn = all.first(where: { $0.transport == kAudioDeviceTransportTypeBuiltIn }) {
            return systemDefault?.id == builtIn.id ? nil : builtIn.id
        }
        if let systemDefault, usable(systemDefault) { return nil }
        return all.first(where: usable)?.id
    }

    static func runSelfTest() -> Bool {
        var passed = true
        func check(_ ok: Bool, _ label: String) {
            if !ok { passed = false }
            print("\(ok ? "PASS" : "FAIL"): microphone: \(label)")
        }
        let builtIn = Device(id: 1, transport: kAudioDeviceTransportTypeBuiltIn)
        let airPods = Device(id: 2, transport: kAudioDeviceTransportTypeBluetooth)
        let studioDisplay = Device(id: 3, transport: kAudioDeviceTransportTypeUSB)
        let teams = Device(id: 4, transport: kAudioDeviceTransportTypeVirtual)
        let everything = [builtIn, airPods, studioDisplay, teams]
        check(choose(systemDefault: airPods, all: everything, builtInDisplayActive: true) == 1,
              "AirPods as the default, lid open: the built-in mic, as before")
        check(choose(systemDefault: airPods, all: everything, builtInDisplayActive: true, prefersBluetooth: true) == nil
              && choose(systemDefault: builtIn, all: everything, builtInDisplayActive: true, prefersBluetooth: true) == 2,
              "AirPods microphone switched on: the headset, whether or not it is the system default")
        check(choose(systemDefault: builtIn, all: [builtIn, studioDisplay], builtInDisplayActive: true, prefersBluetooth: true) == nil,
              "…and with no headset connected, the usual choice")
        check(choose(systemDefault: studioDisplay, all: everything, builtInDisplayActive: true) == 1,
              "lid open: the built-in mic even when a display mic is the default — it starts 0.4 s sooner")
        check(choose(systemDefault: builtIn, all: everything, builtInDisplayActive: true) == nil,
              "the built-in mic as the default, lid open: kept")
        check(choose(systemDefault: builtIn, all: everything, builtInDisplayActive: false) == 3,
              "lid closed: never the disconnected built-in mic — the display's")
        check(choose(systemDefault: airPods, all: everything, builtInDisplayActive: false) == 3,
              "AirPods, lid closed: the wired display microphone")
        check(choose(systemDefault: teams, all: everything, builtInDisplayActive: true) == 1,
              "a virtual device as the default: the built-in mic")
        check(choose(systemDefault: airPods, all: [builtIn, airPods], builtInDisplayActive: false) == nil,
              "lid closed and nothing wired: the default after all, not a dead mic")
        return passed
    }
}

/// Each dictation's microphone start-up — which kind of microphone, how long until the first audio arrived and until
/// the first sound — the last 200, on this Mac only (`~/Library/Application Support/Murmur/mic_timings.json`). Whether
/// a warmer microphone is ever worth its always-on indicator is decided from these real numbers, not guessed.
enum MicTimingLog {
    struct Entry: Codable {
        let date: Date
        let transport: String
        let firstBufferMs: Int?
        let firstSoundMs: Int?
        let onlyZeros: Bool
    }

    static let limit = 200
    private static let queue = DispatchQueue(label: "local.murmur.mic-timing-log")
    static var fileURL: URL { AppPaths.supportDirectory.appendingPathComponent("mic_timings.json") }

    static func record(transport: String, firstBuffer: Int?, firstSound: Int?, onlyZeros: Bool) {
        let entry = Entry(date: Date(), transport: transport, firstBufferMs: firstBuffer, firstSoundMs: firstSound,
                          onlyZeros: onlyZeros)
        queue.async {
            var entries = load()
            entries.append(entry)
            if entries.count > limit { entries.removeFirst(entries.count - limit) }
            if let data = try? JSONEncoder().encode(entries) { try? data.write(to: fileURL, options: .atomic) }
        }
    }

    static func load() -> [Entry] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return (try? JSONDecoder().decode([Entry].self, from: data)) ?? []
    }
}

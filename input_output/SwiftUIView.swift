import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

enum AppScreenState {
    case waiting
    case analysing
    case results
}

struct ContentView: View {
    @State private var screenState: AppScreenState = .waiting
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var isShowingCamera = false
    @State private var statusSubtitle = "Waiting for video"
    @State private var behaviorResult = ""
    @State private var descriptionResult = ""
    @State private var pollingTimer: Timer? = nil
    
    // Change this to your Wi-Fi's IP address as required
    let serverBaseURL = "http://192.168.1.79:5001"
    
    var body: some View {
        VStack(spacing: 28) {
            Spacer()
            
            // MARK: - Screen 1: Waiting for Video
            if screenState == .waiting {
                Text("PET BEHAVIOUR\nANALYSER")
                    .font(.title)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                Text(statusSubtitle)
                    .font(.headline)
                    .foregroundColor(.gray)
                
                VStack(spacing: 16) {
                    PhotosPicker(selection: $selectedItem, matching: .videos) {
                        Text("Video")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(width: 220)
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .onChange(of: selectedItem) { newItem in
                        handlePhotoLibrarySelection(item: newItem)
                    }
                    
                    Button(action: {
                        isShowingCamera = true
                    }) {
                        Text("Record")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(width: 220)
                            .background(Color.orange)
                            .cornerRadius(10)
                    }
                }
            }
            
            // MARK: - Screen 2: Analysing (No Cancel Button)
            else if screenState == .analysing {
                Text("PET BEHAVIOUR\nANALYSER")
                    .font(.title)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                VStack(spacing: 12) {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("ANALYSING...")
                        .font(.title2)
                        .fontWeight(.semibold)
                    Text(statusSubtitle)
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .padding(.vertical, 10)
            }
            
            // MARK: - Screen 3: Analysis Results + Back
            else if screenState == .results {
                Text("ANALYSIS")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                VStack(spacing: 10) {
                    if !behaviorResult.isEmpty {
                        Text(behaviorResult)
                            .font(.title2)
                            .fontWeight(.heavy)
                            .foregroundColor(behaviorResult == "REJECTED" || behaviorResult == "ERROR" ? .red : .green)
                    }
                    
                    Text(descriptionResult)
                        .font(.body)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                VStack(spacing: 6) {
                    Text("Please check TinyScreen for more analysis info.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    Text("Click Back to start a new analysis.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal)
                
                Button(action: {
                    resetSession()
                }) {
                    Text("BACK")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(width: 220)
                        .background(Color.blue)
                        .cornerRadius(10)
                }
            }
            
            Spacer()
        }
        .padding()
        .sheet(isPresented: $isShowingCamera) {
            CameraView(
                isPresented: $isShowingCamera,
                uploadStatus: $statusSubtitle,
                onVideoRecorded: { rawVideoURL in
                    processLiveVideo(inputURL: rawVideoURL)
                }
            )
        }
        .onAppear {
            startPollingForResults()
        }
    }
    
    // MARK: - Session Reset Logic
    
    func resetSession() {
        selectedItem = nil
        behaviorResult = ""
        descriptionResult = ""
        statusSubtitle = "Waiting for video"
        screenState = .waiting
        
        // Notify Python server & TinyScreen+ to reset
        guard let url = URL(string: "\(serverBaseURL)/api/reset") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        URLSession.shared.dataTask(with: request).resume()
    }
    
    // MARK: - Video Processing Logic
    
    func handlePhotoLibrarySelection(item: PhotosPickerItem?) {
        guard let item = item else { return }
        screenState = .analysing
        statusSubtitle = "Loading from library..."
        
        item.loadTransferable(type: VideoFile.self) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let videoFile?):
                    self.statusSubtitle = "Transcoding video to MP4..."
                    self.transcodeToMP4(inputURL: videoFile.url) { newMP4URL in
                        DispatchQueue.main.async {
                            if let finalMP4 = newMP4URL {
                                self.uploadVideoToServer(videoURL: finalMP4)
                            } else {
                                self.statusSubtitle = "Transcoding failed."
                                self.screenState = .waiting
                            }
                        }
                    }
                case .failure(let error):
                    self.statusSubtitle = "Failed to load: \(error.localizedDescription)"
                    self.screenState = .waiting
                default:
                    self.statusSubtitle = "Failed to load video."
                    self.screenState = .waiting
                }
            }
        }
    }
    
    func processLiveVideo(inputURL: URL) {
        screenState = .analysing
        statusSubtitle = "Transcoding to MP4..."
        
        transcodeToMP4(inputURL: inputURL) { newMP4URL in
            DispatchQueue.main.async {
                if let finalMP4 = newMP4URL {
                    self.uploadVideoToServer(videoURL: finalMP4)
                } else {
                    self.statusSubtitle = "Transcoding failed."
                    self.screenState = .waiting
                }
            }
        }
    }
    
    // MARK: - Networking Logic
    
    func uploadVideoToServer(videoURL: URL) {
        statusSubtitle = "Uploading to server..."
        behaviorResult = ""
        descriptionResult = ""
        
        guard let url = URL(string: "\(serverBaseURL)/api/ingest_video") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var data = Data()
        data.append("--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"video\"; filename=\"ios_cat_video.mp4\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
        
        if let videoData = try? Data(contentsOf: videoURL) {
            data.append(videoData)
        }
        
        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        URLSession.shared.uploadTask(with: request, from: data) { responseData, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.statusSubtitle = "Network Error: \(error.localizedDescription)"
                    self.screenState = .waiting
                    return
                }
                self.statusSubtitle = "Processing on server..."
            }
        }.resume()
    }
    
    func startPollingForResults() {
        pollingTimer?.invalidate()
        
        pollingTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
            guard let url = URL(string: "\(serverBaseURL)/api/get_analysis") else { return }
            
            URLSession.shared.dataTask(with: url) { data, response, error in
                guard let data = data, error == nil else { return }
                
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let status = json["status"] as? String {
                    
                    DispatchQueue.main.async {
                        if status == "complete" || status == "error" {
                            if self.screenState == .analysing || self.screenState == .results {
                                self.behaviorResult = json["behavior"] as? String ?? ""
                                self.descriptionResult = json["description"] as? String ?? ""
                                self.screenState = .results
                            }
                        } else if status == "waiting" {
                            // Syncs if TinyScreen+ physical reset button was pressed
                            if self.screenState == .results {
                                self.behaviorResult = ""
                                self.descriptionResult = ""
                                self.selectedItem = nil
                                self.statusSubtitle = "Waiting for video"
                                self.screenState = .waiting
                            }
                        }
                    }
                }
            }.resume()
        }
    }
    
    // MARK: - Background Transcoder
    
    func transcodeToMP4(inputURL: URL, completion: @escaping (URL?) -> Void) {
        let asset = AVAsset(url: inputURL)
        let tempDirectory = FileManager.default.temporaryDirectory
        let outputURL = tempDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        
        guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            completion(nil)
            return
        }
        
        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mp4
        exportSession.shouldOptimizeForNetworkUse = true
        
        exportSession.exportAsynchronously {
            if exportSession.status == .completed {
                completion(outputURL)
            } else {
                completion(nil)
            }
        }
    }
}

// MARK: - Camera Integration

struct CameraView: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    @Binding var uploadStatus: String
    var onVideoRecorded: (URL) -> Void
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera
        picker.mediaTypes = [UTType.movie.identifier]
        picker.videoQuality = .typeHigh
        picker.cameraCaptureMode = .video
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraView
        
        init(_ parent: CameraView) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let mediaURL = info[.mediaURL] as? URL {
                parent.onVideoRecorded(mediaURL)
            }
            parent.isPresented = false
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.isPresented = false
            parent.uploadStatus = "Waiting for video"
        }
    }
}

// MARK: - Custom Transferable Struct

struct VideoFile: Transferable {
    let url: URL
    
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let copyURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("mp4")
            
            try FileManager.default.copyItem(at: received.file, to: copyURL)
            return VideoFile(url: copyURL)
        }
    }
}

#Preview {
    ContentView()
}

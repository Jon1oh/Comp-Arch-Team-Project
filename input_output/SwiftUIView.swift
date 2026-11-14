import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var isShowingCamera = false
    @State private var uploadStatus = "Waiting for video..."
    @State private var behaviorResult = ""
    @State private var descriptionResult = ""
    @State private var pollingTimer: Timer? = nil
    
    // Your Mac's static local network IP
    let serverBaseURL = "http://192.168.1.79:5001"
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Cat Behavior AI") //change title of app here
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // Output Display
            VStack(spacing: 10) {
                Text(uploadStatus)
                    .foregroundColor(.gray)
                
                if !behaviorResult.isEmpty {
                    Text(behaviorResult)
                        .font(.title)
                        .foregroundColor(behaviorResult == "REJECTED" || behaviorResult == "ERROR" ? .red : .green)
                        .fontWeight(.heavy)
                    
                    Text(descriptionResult)
                        .font(.body)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }
            .frame(height: 150)
            
            // Button 1: Upload from Library
            PhotosPicker(selection: $selectedItem, matching: .videos) {
                Text("Select Video")
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
            
            // Button 2: Record Live
            Button(action: {
                isShowingCamera = true
            }) {
                Text("Record Live Video")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(width: 220)
                    .background(Color.orange)
                    .cornerRadius(10)
            }
            .sheet(isPresented: $isShowingCamera) {
                CameraView(
                    isPresented: $isShowingCamera,
                    uploadStatus: $uploadStatus,
                    onVideoRecorded: { rawVideoURL in
                        processLiveVideo(inputURL: rawVideoURL)
                    }
                )
            }
        }
        .padding()
    }
    
    // MARK: - Video Processing Logic
    
    func handlePhotoLibrarySelection(item: PhotosPickerItem?) {
        guard let item = item else { return }
        uploadStatus = "Loading from library..."
        
        item.loadTransferable(type: VideoFile.self) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let videoFile?):
                    // Update UI and route the library file through the transcoder
                    self.uploadStatus = "Transcoding library video to MP4..."
                    
                    self.transcodeToMP4(inputURL: videoFile.url) { newMP4URL in
                        DispatchQueue.main.async {
                            if let finalMP4 = newMP4URL {
                                self.uploadVideoToServer(videoURL: finalMP4)
                            } else {
                                self.uploadStatus = "Transcoding failed. Upload aborted."
                            }
                        }
                    }
                    
                case .failure(let error):
                    self.uploadStatus = "Failed to load video: \(error.localizedDescription)"
                default:
                    self.uploadStatus = "Failed to load video."
                }
            }
        }
    }
    
    func processLiveVideo(inputURL: URL) {
        uploadStatus = "Transcoding to MP4..."
        
        transcodeToMP4(inputURL: inputURL) { newMP4URL in
            DispatchQueue.main.async {
                if let finalMP4 = newMP4URL {
                    self.uploadVideoToServer(videoURL: finalMP4)
                } else {
                    self.uploadStatus = "Transcoding failed. Upload aborted."
                }
            }
        }
    }
    
    // MARK: - Networking Logic
    
    func uploadVideoToServer(videoURL: URL) {
        uploadStatus = "Uploading to server..."
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
                    self.uploadStatus = "Network Error: \(error.localizedDescription)"
                    return
                }
                self.uploadStatus = "Processing on server..."
                self.startPollingForResults()
            }
        }.resume()
    }
    
    func startPollingForResults() {
        pollingTimer?.invalidate()
        
        pollingTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { _ in
            guard let url = URL(string: "\(serverBaseURL)/api/get_analysis") else { return }
            
            URLSession.shared.dataTask(with: url) { data, response, error in
                guard let data = data, error == nil else { return }
                
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let status = json["status"] as? String {
                    
                    DispatchQueue.main.async {
                        if status == "complete" || status == "error" {
                            self.behaviorResult = json["behavior"] as? String ?? ""
                            self.descriptionResult = json["description"] as? String ?? ""
                            self.uploadStatus = (status == "complete") ? "Analysis Complete!" : "Pipeline Error"
                            self.pollingTimer?.invalidate() // Stop asking once we have a result
                        } else if status == "waiting" {
                            // Clear UI if physical button on TinyScreen dismissed it
                            self.behaviorResult = ""
                            self.descriptionResult = ""
                            self.uploadStatus = "Waiting for video..."
                            self.pollingTimer?.invalidate()
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
            parent.uploadStatus = "Waiting for video..."
        }
    }
}

// MARK: - Custom Transferable Struct
// Translates the locked photo library file into a usable file for the app
struct VideoFile: Transferable {
    let url: URL
    
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            // Create a safe temporary path for the app to use
            let copyURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("mp4")
            
            // Copy the video from the locked library into the app
            try FileManager.default.copyItem(at: received.file, to: copyURL)
            return VideoFile(url: copyURL)
        }
    }
}

#Preview {
    ContentView()
}
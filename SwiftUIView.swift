import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var uploadStatus: String = "Waiting for video..."
    @State private var isUploading: Bool = false
    
    // Live camera states
    @State private var showCamera: Bool = false
    @State private var capturedVideoURL: URL? = nil
    
    // REPLACE WITH YOUR MACBOOK'S WI-FI IP ADDRESS
    let serverURL = "http://192.168.1.79:5001/api/ingest_video"

    var body: some View {
        VStack(spacing: 20) {
            Text("Cat Behavior Analyzer")
                .font(.title)
                .bold()
            
            // 1. Existing Library Picker
            PhotosPicker(selection: $selectedItem, matching: .videos) {
                Text("Select Cat Video")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(width: 220)
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .task(id: selectedItem) {
                if let item = selectedItem {
                    await handleLibrarySelection(item: item)
                }
            }
            
            // 2. New Live Camera Button
            Button(action: {
                showCamera = true
            }) {
                Text("Record Live Video")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(width: 220)
                    .background(Color.red)
                    .cornerRadius(10)
            }
            .sheet(isPresented: $showCamera) {
                CameraView(videoURL: $capturedVideoURL)
            }
            .task(id: capturedVideoURL) {
                if let url = capturedVideoURL {
                    handleLiveVideo(url: url)
                }
            }
            
            if isUploading {
                ProgressView("Uploading to Pipeline...")
                    .padding(.top, 20)
            }
            
            Text(uploadStatus)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding()
        }
        .padding()
    }
    
    // --- Handlers ---
    
    private func handleLibrarySelection(item: PhotosPickerItem) async {
        uploadStatus = "Preparing library video..."
        do {
            if let videoData = try await item.loadTransferable(type: Data.self) {
                uploadVideoToServer(videoData: videoData)
            } else {
                uploadStatus = "Error: Could not read video file."
            }
        } catch {
            uploadStatus = "Error loading video: \(error.localizedDescription)"
        }
    }
    
    private func handleLiveVideo(url: URL) {
        uploadStatus = "Preparing live video..."
        do {
            let videoData = try Data(contentsOf: url)
            uploadVideoToServer(videoData: videoData)
        } catch {
            uploadStatus = "Error reading live video: \(error.localizedDescription)"
        }
    }
    
    // --- Network Upload ---
    
    private func uploadVideoToServer(videoData: Data) {
        guard let url = URL(string: serverURL) else {
            uploadStatus = "Invalid Server URL"
            return
        }
        
        isUploading = true
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"ios_cat_video.mp4\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: video/mp4\r\n\r\n".data(using: .utf8)!)
        body.append(videoData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isUploading = false
                
                if let error = error {
                    self.uploadStatus = "Network Error: \(error.localizedDescription)"
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        self.uploadStatus = "Success! Video handed off to DLC pipeline."
                    } else {
                        self.uploadStatus = "Server rejected file (Code: \(httpResponse.statusCode))"
                    }
                }
            }
        }.resume()
    }
}

// --- Camera Wrapper for SwiftUI ---
struct CameraView: UIViewControllerRepresentable {
    @Binding var videoURL: URL?
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        
        // Ensure the device actually has a camera
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            picker.sourceType = .camera
            picker.mediaTypes = [UTType.movie.identifier]
            picker.videoQuality = .typeHigh
        }
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
            if let url = info[.mediaURL] as? URL {
                parent.videoURL = url
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

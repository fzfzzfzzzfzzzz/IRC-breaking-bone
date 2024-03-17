//
//  ContentView.swift
//  IDRCDemo
//
//  Created by Freddie Nicholson on 11/03/2024.
//

import SwiftUI
import PhotosUI

private struct xrayresult: Identifiable {
     var id = UUID()
     var img: Image
     var classification: String
     var confidence_score: Float
 }


struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: Image?
    @State private var alpha = 50.0
    @State private var classifierUI = "?"
    @State private var confidenceUI = "?"
    @State private var loading = false
    @State private var gradCam: Data?
    @State private var uploads: [xrayresult] = []

    var body: some View {
        HStack() {
            VStack() {
                Text(" Uploads").font(/*@START_MENU_TOKEN@*/.title/*@END_MENU_TOKEN@*/).fontWeight(/*@START_MENU_TOKEN@*/.bold/*@END_MENU_TOKEN@*/)
                ScrollView() {
                    ForEach(0 ..< uploads.count,id: \.self) { value in
                        VStack() {
                            Text("Sample \(value+1)")
                            uploads[value].img.resizable()
                                .aspectRatio(contentMode: .fit).frame(width:200, height:100)
                            Text("\(String(format: "%.2f",uploads[value].confidence_score))% - \(uploads[value].classification)")
                        }.padding(20).background(RoundedRectangle(cornerRadius: 10)
                            .fill(Color.white))
                    }
                }
                
                
            }.padding(20).frame(maxHeight: .infinity ).background(Color(UIColor.systemGray6))
            Spacer()
            VStack() {
                HStack() {
                    Spacer()
                    VStack() {
                        HStack() {
                            if let image = selectedImage {
                                selectedImage?
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 300, height: 300)
                            } else {
                                Text("No image selected")
                            }
                            if let gradCamData = gradCam, let uiImage = UIImage(data: gradCamData) {
                                Image(uiImage: uiImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: 300, height: 300)
                            }
                        }
                        HStack() {
                        Text("Alpha Adjust")
                        Slider(
                            value: $alpha,
                            in: 0...100,
                            onEditingChanged: { editing in
                                print("editing")
                            }
                        ).frame(width:400)
                        }

                    }
                    Spacer()
                }
                if(!loading) {
                VStack() {
                    Text("\(confidenceUI)%").font(.title).fontWeight(/*@START_MENU_TOKEN@*/.bold/*@END_MENU_TOKEN@*/)
                    Text("Confidence Score")
                    Text("Classification **\(classifierUI)**")
                    HStack() {
                        PhotosPicker("Select X-Ray", selection: $selectedItem, matching: .images).onChange(of: selectedItem) {
                            Task {
                                if let loaded = try? await selectedItem?.loadTransferable(type: Image.self) {
                                    selectedImage = loaded
                                    let url = URL(string: "http://localhost:8000/ipad?alpha=\(alpha)")
                                    var request = URLRequest(url: url!)
                                    request.httpMethod = "POST"
                                    loading = true
                                    if let data = try? await selectedItem?.loadTransferable(type: Data.self) {
                                        let imageui = UIImage(data:data)
                                        let b64img = imageui?.jpegData(compressionQuality: 1)?.base64EncodedString() ?? ""
                                        let parameters = ["alpha": alpha, "image": b64img]
                                        if let jsonData = try? JSONSerialization.data(withJSONObject: parameters, options: []) {
                                            
                                            request.httpMethod = "POST"
                                            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                                            request.httpBody = jsonData
                                        }
                                        
                                        let task = URLSession.shared.dataTask(with: request) {(data, response, error) in
                                            struct XrayResponse: Decodable {
                                                let classifier: String
                                                let img: String
                                                let conf: Float
                                            }
                                            print(data, response)
                                            let response = try! JSONDecoder().decode(XrayResponse.self, from: data!)
                                            classifierUI = response.classifier
                                            print(response.img)
                                            print(response.conf)
                                            if let selectedImage {
                                                uploads.append(xrayresult(img: selectedImage, classification: response.classifier, confidence_score: response.conf))
                                            }
                                            confidenceUI = "\(response.conf)"
                                            loading = false
                                            if let data = Data(base64Encoded: response.img ,options: .ignoreUnknownCharacters){
                                                        gradCam = data
                                                        print("works")
                                                print(data)
                                            }
                                        }
                                        task.resume()
                                    } else {
                                        print("Failed")
                                    }
                                }
                            }
                        }
                        }
                    }
                } else {
                    ProgressView()
                }
            }
        }
    }
}


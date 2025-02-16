// swift-tools-version: 6.0
import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath: String = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
    let cuLibPath = "-L\(cuPath)\\lib\\x64"
    let cuIncludePath = "-I\(cuPath)\\include"
    
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuLibPath = "-L\(cuPath)/lib64"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif

let package = Package(
    name: "SwiftCUBLASLt",
    products: [
        .library(
            name: "SwiftCUBLASLt",
            targets: ["SwiftCUBLASLt"]),
        .library(
            name: "cxxCUBLASLt",
            targets: ["cxxCUBLASLt"]),
    ],
    dependencies: 
    [
        .package(url: "https://github.com/machineko/SwiftCU", branch: "main"),
        .package(url: "https://github.com/machineko/SwiftCUBLAS", branch: "main")
    ],
    targets: [
        .target(
            name: "cxxCUBLASLt",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath(cuIncludePath)
            ],
            linkerSettings: [
                .unsafeFlags([
                    cuLibPath,
                ]),
                .linkedLibrary("cublas"),
                .linkedLibrary("cublasLt"),
            ]
        ),
        .target(
            name: "SwiftCUBLASLt",
            dependencies: [
                "cxxCUBLASLt",
                .product(name: "SwiftCU", package: "SwiftCU"),
                .product(name: "cxxCU", package: "SwiftCU"),
                .product(name: "SwiftCUBLAS", package: "SwiftCUBLAS")

            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(
                    [cuIncludePath]
                )
            ]
        ),
          .testTarget(
            name: "SwiftCUBLASLtTests",
           
            dependencies: [
                "cxxCUBLASLt", "SwiftCUBLASLt",
                .product(name: "SwiftCU", package: "SwiftCU"),
                .product(name: "cxxCU", package: "SwiftCU"),
                .product(name: "SwiftCUBLAS", package: "SwiftCUBLAS")
            ],
             swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)
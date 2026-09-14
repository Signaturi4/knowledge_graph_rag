// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DagKBViewer",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "DagKBViewer",
            path: "Sources/DagKBViewer"
        )
    ]
)

import AppKit
import SwiftUI

@main
struct DagKBViewerApp: App {
    // Running as a bare (unbundled) executable, AppKit sometimes leaves the
    // activation policy in a state where the window never gets focus/raises.
    // Force it explicitly so the window reliably comes to the front.
    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Knowledge Graph") {
            ContentView()
                .onAppear {
                    NSApplication.shared.activate(ignoringOtherApps: true)
                }
        }
        .windowResizability(.contentSize)
        .defaultSize(width: 1100, height: 760)
    }
}

/// One simple page: a URL bar for the backend's live graph view, a reload
/// button, an auto-refresh toggle, and the graph itself (server-rendered
/// vis-network.js via WKWebView -- see eval/live_view.py on the backend).
struct ContentView: View {
    @State private var urlString = "http://127.0.0.1:8078/"
    @State private var reloadTick = 0
    @State private var autoRefresh = true
    @State private var everySeconds: Double = 8

    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    @State private var secondsSinceReload = 0.0

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                TextField("Backend graph URL", text: $urlString)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { reload() }

                Button("Go") { reload() }

                Divider().frame(height: 18)

                Toggle("Auto-refresh", isOn: $autoRefresh)
                Stepper(value: $everySeconds, in: 2...60, step: 1) {
                    Text("every \(Int(everySeconds))s")
                }
                .frame(width: 150)

                Button {
                    reload()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .keyboardShortcut("r", modifiers: .command)
            }
            .padding(10)

            Divider()

            GraphWebView(urlString: $urlString, reloadTick: $reloadTick)
        }
        .onReceive(timer) { _ in
            guard autoRefresh else { return }
            secondsSinceReload += 1
            if secondsSinceReload >= everySeconds {
                secondsSinceReload = 0
                reload()
            }
        }
    }

    private func reload() {
        reloadTick += 1
        secondsSinceReload = 0
    }
}

import SwiftUI
import WebKit

/// Thin bridge to WKWebView. The graph itself (vis-network.js, same library
/// family graphify's HTML export uses) is rendered entirely server-side by
/// viz/pyvis_render.py -- this view just displays that page and reloads it.
struct GraphWebView: NSViewRepresentable {
    @Binding var urlString: String
    @Binding var reloadTick: Int

    final class Coordinator {
        var lastURL: String = ""
        var lastTick: Int = -1
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        WKWebView()
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        let urlChanged = context.coordinator.lastURL != urlString
        let tickChanged = context.coordinator.lastTick != reloadTick
        guard urlChanged || tickChanged else { return }
        context.coordinator.lastURL = urlString
        context.coordinator.lastTick = reloadTick
        guard let url = URL(string: urlString) else { return }
        view.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
    }
}

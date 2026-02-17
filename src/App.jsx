import React, { useEffect, useMemo, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import "./App.css";

// Configure worker (Vite-friendly)
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

/**
 * GitHub Pages-safe asset path builder.
 * BASE_URL is "/robotic_controller.github.io/" on GH Pages and "/" in dev.
 * Accepts: "/assets/x.pdf" or "assets/x.pdf"
 * Returns: "/robotic_controller.github.io/assets/x.pdf"
 */
function assetUrl(path) {
  const base = (import.meta.env.BASE_URL || "/").replace(/\/+$/, "/");
  const clean = String(path || "").replace(/^\/+/, "");
  return base + clean;
}

/** ErrorBoundary so you never get a blank screen again */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    console.error("App crashed:", error, info);
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 20, fontFamily: "system-ui" }}>
          <h2 style={{ margin: 0 }}>Page crashed</h2>
          <p style={{ opacity: 0.8 }}>
            Open DevTools → Console to see the full error. Here is the message:
          </p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              background: "#111",
              color: "#fff",
              padding: 12,
              borderRadius: 12,
              overflow: "auto",
            }}
          >
            {String(this.state.error?.message || this.state.error)}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const CONFIG = {
  title: "Test-Driven Agentic Framework for Reliable Robot Controller",
  venueLine: "Submitted to IROS 2026",
  authors: [
    { name: "Shivanshu Tripathi", aff: "1" },
    { name: "Reza Akbarian Bafghi", aff: "2" },
    { name: "Maziar Raissi", aff: "1" },
  ],
  affiliations: [
    { id: "1", text: "UC Riverside" },
    { id: "2", text: "CU Boulder" },
  ],
  links: [
    { label: "Paper", href: "https://arxiv.org/abs/XXXX.XXXXX" },
    { label: "Video", href: "https://youtu.be/XXXXXXXXXXX" },
    { label: "Code", href: "https://github.com/your/repo" },
  ],
  abstract:
    "We present a test-driven, agentic framework for synthesizing a deployable low-level robot controller for navigation. " +
    "Given either a 2D navigation map with robot constraints or a 3D simulation environment, our framework iteratively refines " +
    "the generated controller code using diagnostic feedback from structured test suites. We evaluate the approach across 2D navigation " +
    "tasks and 3D navigation in the Webots simulator, showing substantial improvements in reliability and robustness over one-shot prompting.",

  videos: [
    {
      title: "2D Navigation Demo",
      desc: "Map-based navigation with test-driven repair (2D).",
      media: { type: "youtube", id: "dQw4w9WgXcQ", title: "2D Video" },
      tags: ["2D", "map", "tests"],
    },
    {
      title: "Webots Demo",
      desc: "3D navigation in Webots with e-puck robot controller synthesis.",
      media: { type: "youtube", id: "dQw4w9WgXcQ", title: "Webots Video" },
      tags: ["3D", "webots", "e-puck"],
    },
  ],

  // Use your real file:
  methodologyFigures: [
    {
      title: "Agentic Workflow",
      caption: "Test-driven synthesis loop: generate → test → diagnose → repair.",
      src: "/assets/PyTest1.pdf",
    },
  ],

  plotCards: [
    {
      title: "2D Navigation Results",
      subtitle: "Map-based navigation benchmarks",
      bullets: ["Success rate and cumulative success (combined)"],
      plots: [{ label: "2D: SR + CS", src: "/assets/success_rate.pdf" }], // if combined, point to combined file
      // If you actually want separate: use plot_2d_sr.pdf and plot_2d_cs.pdf as two entries.
    },
    {
      title: "Webots Results",
      subtitle: "3D simulation benchmarks",
      bullets: ["Success rate and cumulative success (combined)"],
      plots: [{ label: "Webots: SR + CS", src: "/assets/comp_webots.pdf" }], // if combined, point to combined file
    },
  ],

  footer: {
    contact: "strip008@ucr.edu",
    copyright: "© 2026",
  },
};

const SECTIONS = [
  { id: "abstract", label: "Abstract" },
  { id: "videos", label: "Videos" },
  { id: "methodology", label: "Methodology" },
  { id: "plots", label: "Plots" },
];

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

function useActiveSection(sectionIds, offsetPx = 96) {
  const [active, setActive] = useState(sectionIds[0] || "");

  useEffect(() => {
    const els = sectionIds.map((id) => document.getElementById(id)).filter(Boolean);
    if (els.length === 0) return;

    const onScroll = () => {
      const y = window.scrollY + offsetPx + 1;
      let best = els[0].id;
      for (const el of els) {
        if (el.offsetTop <= y) best = el.id;
      }
      setActive(best);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [sectionIds, offsetPx]);

  return active;
}

function scrollToId(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const top = el.getBoundingClientRect().top + window.scrollY - 84;
  window.scrollTo({ top, behavior: "smooth" });
}

function Pill({ children }) {
  return <span className="pill">{children}</span>;
}

function ButtonLink({ href, children, variant = "primary" }) {
  return (
    <a
      className={classNames("btn", variant === "secondary" && "btnSecondary")}
      href={href}
      target="_blank"
      rel="noreferrer"
    >
      {children}
    </a>
  );
}

function VideoFrame({ media }) {
  if (!media) return null;

  if (media.type === "youtube") {
    const src = `https://www.youtube-nocookie.com/embed/${media.id}`;
    return (
      <div className="videoFrame" aria-label={media.title || "Video"}>
        <iframe
          src={src}
          title={media.title || "YouTube video"}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    );
  }

  if (media.type === "mp4") {
    return (
      <div className="videoFrame" aria-label={media.title || "Video"}>
        <video controls playsInline>
          <source src={media.src} type="video/mp4" />
        </video>
      </div>
    );
  }

  return null;
}

/**
 * Render first page of a PDF into a canvas and auto-crop whitespace.
 * This makes the figure appear at its "actual" size instead of centered on a huge PDF page.
 */
function PdfCropped({ src, className = "", maxScale = 3, pad = 14 }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let cancelled = false;
    let ro = null;

    async function render() {
      try {
        setStatus("loading");
        const wrap = wrapRef.current;
        const canvas = canvasRef.current;
        if (!wrap || !canvas) return;

        const url = assetUrl(src);

        const loadingTask = pdfjsLib.getDocument({ url });
        const pdf = await loadingTask.promise;
        if (cancelled) return;

        const page = await pdf.getPage(1);
        if (cancelled) return;

        const wrapW = Math.max(320, wrap.clientWidth);
        const viewport1 = page.getViewport({ scale: 1 });
        const fitScale = Math.min(maxScale, wrapW / viewport1.width);
        const viewport = page.getViewport({ scale: fitScale });

        const off = document.createElement("canvas");
        const offCtx = off.getContext("2d", { willReadFrequently: true });

        off.width = Math.ceil(viewport.width);
        off.height = Math.ceil(viewport.height);

        await page
          .render({
            canvasContext: offCtx,
            viewport,
            background: "white",
          })
          .promise;

        if (cancelled) return;

        const img = offCtx.getImageData(0, 0, off.width, off.height);
        const { data, width, height } = img;

        const isContent = (i) => {
          const r = data[i],
            g = data[i + 1],
            b = data[i + 2],
            a = data[i + 3];
          if (a === 0) return false;
          // near-white threshold
          return !(r > 245 && g > 245 && b > 245);
        };

        let minX = width,
          minY = height,
          maxX = -1,
          maxY = -1;

        for (let y = 0; y < height; y++) {
          const row = y * width * 4;
          for (let x = 0; x < width; x++) {
            const i = row + x * 4;
            if (isContent(i)) {
              if (x < minX) minX = x;
              if (y < minY) minY = y;
              if (x > maxX) maxX = x;
              if (y > maxY) maxY = y;
            }
          }
        }

        if (maxX < 0 || maxY < 0) {
          minX = 0;
          minY = 0;
          maxX = width - 1;
          maxY = height - 1;
        }

        minX = Math.max(0, minX - pad);
        minY = Math.max(0, minY - pad);
        maxX = Math.min(width - 1, maxX + pad);
        maxY = Math.min(height - 1, maxY + pad);

        const cropW = Math.max(1, maxX - minX + 1);
        const cropH = Math.max(1, maxY - minY + 1);

        const ctx = canvas.getContext("2d");
        canvas.width = cropW;
        canvas.height = cropH;

        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, cropW, cropH);
        ctx.drawImage(off, minX, minY, cropW, cropH, 0, 0, cropW, cropH);

        setStatus("ready");
      } catch (e) {
        console.error("PdfCropped render error:", e);
        setStatus("error");
      }
    }

    render();

    if (wrapRef.current) {
      ro = new ResizeObserver(() => render());
      ro.observe(wrapRef.current);
    }

    return () => {
      cancelled = true;
      if (ro) ro.disconnect();
    };
  }, [src, maxScale, pad]);

  return (
    <div ref={wrapRef} className={classNames("pdfCropWrap", className)}>
      {status !== "ready" ? (
        <div className="pdfCropStatus">
          {status === "loading" ? "Loading figure…" : "Could not render PDF."}
        </div>
      ) : null}
      <canvas ref={canvasRef} className="pdfCropCanvas" />
    </div>
  );
}

function Section({ id, title, children, kicker, underlineTitle = false, centerTitle = false }) {
  return (
    <section id={id} className="section">
      <div className={classNames("sectionHeader", centerTitle && "sectionHeaderCentered")}>
        {kicker ? <div className="kicker">{kicker}</div> : null}
        <h2 className={classNames(underlineTitle && "titleUnderline")}>{title}</h2>
      </div>
      <div>{children}</div>
    </section>
  );
}

function Card({ title, desc, children, tags = [] }) {
  return (
    <div className="card">
      <div className="cardTop">
        <h3>{title}</h3>
        {tags?.length ? (
          <div className="tagRow" aria-label="tags">
            {tags.map((t) => (
              <span key={t} className="tag">
                {t}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {desc ? <p className="muted">{desc}</p> : null}
      {children}
    </div>
  );
}

function FigureCard({ title, caption, src }) {
  const url = assetUrl(src);
  return (
    <div className="figureCard">
      <div className="figureHeader">
        <div className="figureTitle">{title}</div>
        <div className="figureCaption">{caption}</div>
      </div>
      <div className="figureBody">
        <PdfCropped src={src} className="figureCropBig" pad={12} maxScale={3} />
        <div className="figureActions">
          <a className="btn btnSecondary" href={url} target="_blank" rel="noreferrer">
            Open PDF
          </a>
          <a className="btn btnSecondary" href={url} download>
            Download
          </a>
        </div>
      </div>
    </div>
  );
}

function PlotCard({ title, subtitle, bullets, plots }) {
  return (
    <div className="plotCard">
      <div className="plotHeader">
        <div>
          <div className="plotTitle">{title}</div>
          <div className="plotSubtitle">{subtitle}</div>
        </div>
      </div>

      <ul className="plotBullets">
        {bullets.map((b) => (
          <li key={b}>{b}</li>
        ))}
      </ul>

      <div className="plotGridBig">
        {plots.map((p) => {
          const url = assetUrl(p.src);
          return (
            <div className="plotItemBig" key={p.label}>
              <div className="plotLabelRow">
                <div className="plotLabel">{p.label}</div>
                <div className="plotLinks">
                  <a className="miniLink" href={url} target="_blank" rel="noreferrer">
                    Open
                  </a>
                  <a className="miniLink" href={url} download>
                    Download
                  </a>
                </div>
              </div>

              <div className="plotBody">
                <PdfCropped src={p.src} className="plotCropBig" pad={10} maxScale={3} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AppInner() {
  const active = useActiveSection(SECTIONS.map((s) => s.id));
  const topRef = useRef(null);

  const authorLine = useMemo(() => {
    return (CONFIG.authors || [])
      .map((a) => `${a.name}\u00A0${a.aff ? `^${a.aff}` : ""}`)
      .join(", ");
  }, []);

  return (
    <div className="page">
      <header className="topbar">
        <div className="topbarInner">
          <button
            className="brand"
            onClick={() => topRef.current?.scrollIntoView({ behavior: "smooth" })}
            aria-label="Scroll to top"
          >
            Home
          </button>

          <nav className="nav" aria-label="Sections">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                className={classNames("navLink", active === s.id && "navLinkActive")}
                onClick={() => scrollToId(s.id)}
              >
                {s.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main ref={topRef} className="heroWrap">
        <div className="hero heroCentered">
          <h1 className="heroTitle">{CONFIG.title}</h1>

          <div className="pillRow">
            {(CONFIG.venueLine || "").split("•").map((x) => (
              <Pill key={x.trim()}>{x.trim()}</Pill>
            ))}
          </div>

          <div className="authors">
            <div className="authorsLine">{authorLine}</div>
            <div className="affiliations">
              {(CONFIG.affiliations || []).map((a) => (
                <div key={a.id} className="aff">
                  <span className="sup">^{a.id}</span> {a.text}
                </div>
              ))}
            </div>
          </div>

          <div className="linkRow">
            {(CONFIG.links || []).map((l) => (
              <ButtonLink key={l.label} href={l.href} variant={l.label === "Paper" ? "primary" : "secondary"}>
                {l.label}
              </ButtonLink>
            ))}
          </div>
        </div>
      </main>

      <div className="content">
        <Section id="abstract" title="Abstract" underlineTitle centerTitle>
          <div className="prose proseCentered">
            <p>{CONFIG.abstract}</p>
          </div>
        </Section>

        <Section id="videos" title="Demos">
          <div className="grid2">
            {(CONFIG.videos || []).map((v) => (
              <Card key={v.title} title={v.title} desc={v.desc} tags={v.tags}>
                <VideoFrame media={v.media} />
              </Card>
            ))}
          </div>
        </Section>

        <Section id="methodology" title="Methodology">
          <div className="grid1">
            {(CONFIG.methodologyFigures || []).map((f) => (
              <FigureCard key={f.title} title={f.title} caption={f.caption} src={f.src} />
            ))}
          </div>
        </Section>

        <Section id="plots" title="Results">
          <div className="grid2">
            {(CONFIG.plotCards || []).map((c) => (
              <PlotCard
                key={c.title}
                title={c.title}
                subtitle={c.subtitle}
                bullets={c.bullets || []}
                plots={c.plots || []}
              />
            ))}
          </div>
        </Section>

        <footer className="footer">
          <div className="footerInner">
            <div>
              <div className="footerTitle">{CONFIG.title}</div>
              <div className="muted small">{CONFIG.footer.copyright}</div>
            </div>
            <div className="footerRight">
              <div className="muted small">Contact</div>
              <div className="footerContact">{CONFIG.footer.contact}</div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppInner />
    </ErrorBoundary>
  );
}

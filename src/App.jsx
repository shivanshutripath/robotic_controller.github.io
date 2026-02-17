import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

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
    "the generated controller code using diagnostic feedback from structured test suites. We propose a dual-tier repair strategy " +
    "that alternates between prompt-level refinement and direct code editing to satisfy interface compliance, safety constraints, " +
    "and task success criteria. We evaluate the approach across 2D navigation tasks and 3D navigation in the Webots simulator. " +
    "Experimental results show that test-driven synthesis substantially improves controller reliability and robustness over one-shot prompting, " +
    "particularly when initial specifications are underspecified or suboptimal.",

  mainVideo: { type: "youtube", id: "dQw4w9WgXcQ", title: "Overview Video" },

  methodCards: [
    {
      title: "2D Navigation (Map-based)",
      desc: "Controller synthesis and iterative repair for occupancy-grid navigation with interface and safety constraints.",
      media: { type: "youtube", id: "dQw4w9WgXcQ", title: "2D demo" },
      tags: ["2D", "tests", "repair"],
    },
    {
      title: "3D Navigation (Webots)",
      desc: "Controller synthesis for an e-puck robot in Webots using sensor-driven execution and goal-reaching tests.",
      media: { type: "youtube", id: "dQw4w9WgXcQ", title: "3D demo" },
      tags: ["3D", "webots", "tests"],
    },
  ],

  footer: {
    contact: "strip008@ucr.edu",
    copyright: "© 2026",
  },
};

const SECTIONS = [
  { id: "abstract", label: "Abstract" },
  { id: "video", label: "Video" },
  { id: "method", label: "Method" },
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

function Section({ id, title, children, kicker, underlineTitle = false, centerTitle = false }) {
  return (
    <section id={id} className="section">
      <div
        className={classNames(
          "sectionHeader",
          centerTitle && "sectionHeaderCentered"
        )}
      >
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

export default function App() {
  const active = useActiveSection(SECTIONS.map((s) => s.id));
  const topRef = useRef(null);

  const authorLine = useMemo(() => {
    return CONFIG.authors.map((a) => `${a.name}\u00A0${a.aff ? `^${a.aff}` : ""}`).join(", ");
  }, []);

  return (
    <div className="page">
      {/* Top Nav */}
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

      {/* Hero */}
      <main ref={topRef} className="heroWrap">
        <div className="hero heroCentered">
          <h1 className="heroTitle">{CONFIG.title}</h1>

          <div className="pillRow" aria-label="highlights">
            {CONFIG.venueLine.split("•").map((x) => (
              <Pill key={x.trim()}>{x.trim()}</Pill>
            ))}
          </div>

          <div className="authors">
            <div className="authorsLine">{authorLine}</div>
            <div className="affiliations">
              {CONFIG.affiliations.map((a) => (
                <div key={a.id} className="aff">
                  <span className="sup">^{a.id}</span> {a.text}
                </div>
              ))}
            </div>
          </div>

          <div className="linkRow">
            {CONFIG.links.map((l) => (
              <ButtonLink
                key={l.label}
                href={l.href}
                variant={l.label === "Paper" ? "primary" : "secondary"}
              >
                {l.label}
              </ButtonLink>
            ))}
          </div>
        </div>
      </main>

      {/* Content */}
      <div className="content">
        <Section
          id="abstract"
          title="Abstract"
          underlineTitle
          centerTitle
        >
          <div className="prose proseCentered">
            <p>{CONFIG.abstract}</p>
          </div>
        </Section>

        <Section id="video" title="Video" kicker="Overview">
          <VideoFrame media={CONFIG.mainVideo} />
        </Section>

        <Section id="method" title="Method" kicker="Key components">
          <div className="grid2">
            {CONFIG.methodCards.map((p) => (
              <Card key={p.title} title={p.title} desc={p.desc} tags={p.tags}>
                <VideoFrame media={p.media} />
              </Card>
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

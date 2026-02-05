//export const PAPERS = {
//     "active-probing-multimodal-predictions": {
//       title: "Your Paper Title",
//       authors: ["You", "Coauthor"],
//       venue: "Conference, 2026",
//       links: {
//         paper: "https://arxiv.org/abs/xxxx",
//         code: "https://github.com/you/repo",
//         video: "https://youtube.com/...",
//       },
//       abstract: "Paste abstract here...",
//       bibtex: `@inproceedings{...}`,
//     },
//   };
import { useParams } from "react-router-dom";

export default function Paper() {
  const { slug } = useParams();
  return (
    <div style={{ padding: 24 }}>
      <h1>Paper: {slug}</h1>
      <p>Add your paper content here.</p>
    </div>
  );
}

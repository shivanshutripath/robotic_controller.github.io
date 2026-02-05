import { useParams } from "react-router-dom";

export default function Paper() {
  const { slug } = useParams();
  return (
    <div style={{ padding: 24 }}>
      <h1>{slug}</h1>
      <p>Paper page template.</p>
    </div>
  );
}

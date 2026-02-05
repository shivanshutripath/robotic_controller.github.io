import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Paper from "./pages/Paper";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/papers/:slug" element={<Paper />} />
    </Routes>
  );
}

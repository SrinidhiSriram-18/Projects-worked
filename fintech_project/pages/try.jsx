import Header from "../components/Header";
import Nav from "../components/NavNormal";
import Footer from "../components/Footer";
import App from "../components/App/App";

export default function Try() {
  const url = "https://is5009-ais.now.sh/";
  const title = "Actionable Intelligence System";
  const ogImage = "/images/logo/logo.png";
  return (
    <div id="app">
      <Header url={url} title={title} ogImage={ogImage} />
      <Nav />
      <br />
      <App />
      <Footer />
    </div>
  );
}

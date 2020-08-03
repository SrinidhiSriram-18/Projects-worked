import Header from '../components/Header';
import Nav from '../components/Nav'
import Hero from '../components/Hero'
import About from '../components/About'
import Footer from '../components/Footer'

const title = 'Actionable Intelligence System 🚀'
const subtitle = 'Make data work for you.'
const description = `AIS is a state-of-the-art smart and intelligent system 
that analyses a bank's transaction data to derive actionable insights that 
can truly empower the business and end-users.`

export default function Index() {
  return (
    <div>
      <Header />
      <Nav />
      <Hero title={title} subtitle={subtitle} description={description} />
      <About />
      <Footer />
    </div>
  );
}

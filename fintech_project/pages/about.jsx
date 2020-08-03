import Head from '../components/Header'
import About from '../components/About'
import Nav from '../components/Nav'
import Hero from '../components/Hero'
import Footer from '../components/Footer'

const title = 'About'
const subtitle = 'Everything you need to know about us'
const description = ''

export default function AboutPage() {
  return (
    <div>
      <Head title="About" />
      <Nav />
      <Hero title={title} subtitle={subtitle} description={description} />
      <About />
      <Footer />
    </div>
  )
}

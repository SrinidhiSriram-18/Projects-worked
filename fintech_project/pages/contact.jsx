import Head from '../components/Header'
import Nav from '../components/Nav'
import Hero from '../components/Hero'
import Contact from '../components/Contact'
import Footer from '../components/Footer'

const title = 'Contact'
const subtitle = 'Got a question. Drop it here!'
const description = ''

export default function ContactForm() {
  return (
    <div>
      <Head title="Contact" />
      <Nav />
      <Hero title={title} subtitle={subtitle} description={description} />
      <Contact />
      <Footer />
    </div>
  )
}

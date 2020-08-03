import React from "react";

const About = () => (
  <div>
    <div className="section" id="about">
      <div className="container">
        <div className="row">
          <div className="col-sm-12">
            <h1 className="section-heading">What is AIS?</h1>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-6">
            <p className="section-description">
              AIS stands for Actionable Intelligence System. AIS is an ingenius
              solution for banks who would like to take full advantage of their
              transaction data. By using Machine Learning models, AIS delivers
              intelligent insights so as to empower banks to power their
              products with smart suggestions and relevant contextual
              notifications.
            </p>
          </div>
          <div className="col-sm-6 text-center">
            <img
              className="step-img img-responsive"
              src="/images/logo/logo.png"
            />
          </div>
        </div>
        <br />
      </div>
    </div>
    <div className="section container" id="about-more">
      <div className="row">
        <div className="col-sm-2">
          <img
            className="step-img img-responsive"
            src="/images/about/persona.png"
          />
        </div>
        <div className="col-sm-2">
          <img
            className="step-img img-responsive"
            src="/images/about/contextual-notification.png"
          />
        </div>
        <div className="col-sm-2">
          <img
            className="step-img img-responsive"
            src="/images/about/lifelong-companion.png"
          />
        </div>
        <div className="col-sm-6">
          <h1 className="section-heading">Data. Data. Data.</h1>
          <p className="section-description">
            Banks have a wealth of data. AIS effectively makes use of data from
            banks to return powerful analytics that allows banks to have{" "}
            <b>Actionable Intelligence</b>. AIS offers 3 main features:
            <ul>
              <li>Personification</li>
              <li>Contextual Notifications</li>
              <li>Lifelong Companion</li>
            </ul>
          </p>
        </div>
      </div>
    </div>
    <div className="section" id="about">
      <div className="container">
        <div className="row">
          <div className="col-sm-6">
            <h1 className="section-heading">
              Free to try. <br />
              Go Pro for power.
            </h1>
            <p className="section-description">
              AIS works on a freemium model. If you are curious to give AIS a
              test-drive, check out the "Try" section <a href="/try">here</a>.
              If you are ready to integrate AIS into your financial
              infrastructure, drop us a note!
            </p>
          </div>
          <div className="col-sm-6">
            <img
              className="step-img img-responsive"
              src="/images/about/app-screen.png"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default About;

import React from "react";

const Hero = props => (
  <div className="hero filter-gradient">
    <div className="text-center">
      <div className="container">
        <div className="row">
          <div className="col-sm-12">
            <span className="cover-brand-title">{props.title}</span>
            <p className="cover-subtitle">{props.subtitle}</p>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-6 offset-md-3">
            <p className="cover-description">{props.description}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default Hero;

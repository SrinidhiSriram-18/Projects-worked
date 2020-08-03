import React, { useState, useEffect } from "react";
import Card from "./Card";

function Home(props) {
  return (
    <div>
      <div className="row">
        <div className="col-sm-12">
          <div className="bank-name-title">
            <b>Bank of NUS</b>
          </div>
          <span className="bank-tagline">
            Powered by Actionable Intelligence System
          </span>
          <p className="welcome-back-text">
            Welcome back, <b>Jackson</b>!
          </p>
        </div>
      </div>
      <div className="row">
        <div className="col-sm-12">
          <Card />
        </div>
      </div>

      <br />
      <div className="row home-top text-center"></div>
      <div className="row section-header">
        <span className="section-header-text">FEATURES</span>
      </div>
      <div className="row app-row">
        <div
          className="col-sm-4 text-center"
          onClick={props.handleTransactClick}
        >
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/transfer.png"
          />
          <span className="app-icon-text">Transact</span>
        </div>
        <div className="col-sm-4 text-center">
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/chart.png"
          />
          <span className="app-icon-text">Trade</span>
        </div>
        <div
          className="col-sm-4 text-center"
          onClick={props.handleHistoryClick}
        >
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/history.png"
          />
          <span className="app-icon-text">History</span>
        </div>
      </div>
      <div className="row app-row">
        <div
          className="col-sm-4 text-center"
          onClick={props.handleProfileClick}
        >
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/profile.png"
          />
          <span className="app-icon-text">Profile</span>
        </div>
        <div
          className="col-sm-4 text-center"
          onClick={props.handleCompanionClick}
        >
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/companion.png"
          />
          <span className="app-icon-text">Companion</span>
        </div>
        <div className="col-sm-4 text-center">
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/coin.png"
          />
          <span className="app-icon-text">Invest</span>
        </div>
      </div>
      <div className="row app-row">
        <div className="col-sm-4 text-center">
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/scan.png"
          />
          <span className="app-icon-text">Scan</span>
        </div>
        <div className="col-sm-4 text-center">
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/faq.png"
          />
          <span className="app-icon-text">FAQ</span>
        </div>
        <div className="col-sm-4 text-center">
          <img
            className="img-fluid mx-auto app-icon"
            src="/images/app/icons/settings.png"
          />
          <span className="app-icon-text">Settings</span>
        </div>
      </div>
    </div>
  );
}

export default Home;

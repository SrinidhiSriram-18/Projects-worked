import React, { useState, useEffect } from "react";
import Home from "./Screens/Home";
import Persona from "./Screens/Persona";
import Transact from "./Screens/Transact";
import RelatedOffers from "./Screens/RelatedOffers";
import TransactionHistory from "./Screens/TransactionHistory";
import Notification from "./Notification";
import Companion from "./Screens/Companion";

const Screens = {
  HOMEPAGE: 1,
  PERSONA: 2,
  TRANSACT: 3,
  RELATEDOFFERS: 4,
  HISTORY: 5,
  COMPANION: 6
};

function Phone(props) {
  const [notification, setNotification] = useState(null);
  const [currScreen, setCurrScreen] = useState(Screens.HOMEPAGE);

  function notificationBubbleTapped() {
    setCurrScreen(Screens.RELATEDOFFERS);
  }

  function notificationTapped() {
    setCurrScreen(Screens.PERSONA);
  }

  function handleCompanionClick() {
    setCurrScreen(Screens.COMPANION);
  }

  function handleProfileClick() {
    setCurrScreen(Screens.PERSONA);
  }

  function handleTransactClick() {
    console.log("Transact button clicked");
    setCurrScreen(Screens.TRANSACT);
  }

  function handleHistoryClick() {
    console.log("History button clicked");
    setCurrScreen(Screens.HISTORY);
  }

  function goBack() {
    setCurrScreen(Screens.HOMEPAGE);
  }

  useEffect(() => {
    if (currScreen !== 1) {
      setNotification(null);
    } else {
      if (
        props.notification !== null &&
        typeof props.notification.message !== undefined
      ) {
        setNotification(props.notification.message);
      }
    }
  });

  return (
    <div>
      <div className="marvel-device iphone-x">
        <div className="notch">
          <div className="camera"></div>
          <div className="speaker"></div>
        </div>
        <div className="top-bar"></div>
        <div className="sleep"></div>
        <div className="bottom-bar"></div>
        <div className="volume"></div>
        <div className="overflow">
          <div className="shadow shadow--tr"></div>
          <div className="shadow shadow--tl"></div>
          <div className="shadow shadow--br"></div>
          <div className="shadow shadow--bl"></div>
        </div>
        <div className="inner-shadow"></div>
        <div id="parent">
          <div className="screen" id="phone-screen">
            <div className="container phone-frame">
              {notification === null ? (
                <span></span>
              ) : (
                <Notification
                  messageBody={props.notification.message}
                  notificationTapped={notificationTapped}
                />
              )}
              <br />
              <br />
              {currScreen === 1 ? (
                <Home
                  handleTransactClick={handleTransactClick}
                  handleHistoryClick={handleHistoryClick}
                  handleProfileClick={handleProfileClick}
                  handleCompanionClick={handleCompanionClick}
                />
              ) : currScreen === 2 ? (
                <Persona goBack={goBack} />
              ) : currScreen === 3 ? (
                <Transact
                  goBack={goBack}
                  notificationBubbleTapped={notificationBubbleTapped}
                />
              ) : currScreen === 4 ? (
                <RelatedOffers goBack={goBack} />
              ) : currScreen === 5 ? (
                <TransactionHistory goBack={goBack} />
              ) : currScreen === 6 ? (
                <Companion goBack={goBack} />
              ) : (
                <Home
                  handleTransactClicked={handleTransactClick}
                  handleHistoryClick={handleHistoryClick}
                  handleProfileClick={handleProfileClick}
                  handleCompanionClick={handleCompanionClick}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Phone;

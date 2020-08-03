import React, { useState, useEffect } from "react";
import axios from "axios";

function Persona(props) {
  const [persona, setPersona] = useState(null);
  const [deal, setDeal] = useState("");
  const [isPersonaSet, togglePersonaFlag] = useState(false);

  useEffect(() => {
    // load persona from localStorage
    const savedPersona = localStorage.getItem("persona");
    console.log(savedPersona);
    if (savedPersona !== null) {
      setPersona(savedPersona);
      togglePersonaFlag(true);
    }

    async function fetchDeal() {
      const personaUrl = "/api/deals?persona=" + savedPersona;
      const response = await axios.get(personaUrl);
      console.log(response.data.deal);
      setDeal(response.data.deal);
    }
    fetchDeal();
  }, []);
  return (
    <div>
      <div className="row">
        <img
          className="nav-arrow"
          src="/images/navigation/back.png"
          onClick={props.goBack}
        ></img>
      </div>
      <h1 className="screen-title">Your Persona</h1>

      {isPersonaSet ? (
        <div>
          <div className="text-center">
            <img
              className="persona-image"
              src={
                deal.personaImageUrl === null
                  ? "/images/personas/foodie.png"
                  : deal.personaImageUrl
              }
            />
            <h3>{persona}</h3>
          </div>
          <p className="default-text gray-text">
            By analysing you history of transactions, AIS has determined that
            you are a <b>{persona}</b>! AIS will now use this classification to
            recommend you more relevant deals.
          </p>
          <div className="deal">
            <div className="row text-center">
              <img className="deal-image" src={deal.imageUrl} />
            </div>
            <h5 className="deal-title">{deal.title}</h5>
            <p className="gray-text deal-message">{deal.message}</p>
            <p className="gray-text deal-message">Tap to claim!</p>
          </div>
          <div className="error-msg text-center">
            <span>Report an error with this classification</span>
          </div>
        </div>
      ) : (
        <div className="default-text gray-text">
          Please upload some transaction data to get started!
        </div>
      )}
    </div>
  );
}

export default Persona;

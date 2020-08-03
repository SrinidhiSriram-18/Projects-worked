import React, { useState } from "react";
import UploadData from "./TestScreens/PersonaClassification";
import ContextualNotifications from "./TestScreens/ContextualNotifications";
import Companion from "./TestScreens/LifelongCompanion";
import LifelongCompanion from "./TestScreens/LifelongCompanion";

function Tester(props) {
  const [tab, setTab] = useState(1);

  function updateNotification(update) {
    console.log(update);
    props.handleUpdates(update);
  }

  return (
    <div className="tester-container">
      {tab === 1 ? (
        <UploadData updateNotification={updateNotification} />
      ) : tab === 2 ? (
        <ContextualNotifications />
      ) : (
        <LifelongCompanion />
      )}
      <div className="btn-group-sm test-buttons" role="group">
        <button
          type="button"
          className="btn btn-secondary test-btn"
          onClick={() => setTab(1)}
        >
          Persona
        </button>
        <button
          type="button"
          className="btn btn-secondary test-btn"
          onClick={() => setTab(2)}
        >
          Contextual Notifications
        </button>
        <button
          type="button"
          className="btn btn-secondary test-btn"
          onClick={() => setTab(3)}
        >
          Lifelong Companion
        </button>
      </div>
    </div>
  );
}

export default Tester;

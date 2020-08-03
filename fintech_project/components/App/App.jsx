import React, { useState } from "react";
import Tester from "./Tester";
import Phone from "./Phone";

function App() {
  const [notification, setNotification] = useState(null);
  function handleUpdates(update) {
    console.log(update);
    setNotification(update);
  }
  return (
    <div className="container">
      <div className="row">
        <div className="col-md-6">
          <Tester handleUpdates={handleUpdates} />
        </div>
        <div className="col-md-6">
          <Phone handleUpdates={handleUpdates} notification={notification} />
        </div>
      </div>
    </div>
  );
}

export default App;

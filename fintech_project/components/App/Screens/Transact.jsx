import React, { useState } from "react";
import NotificationBubble from "../NotificationBubble";

function Transact(props) {
  const [amount, setAmount] = useState(0);
  const [merchant, setMerchant] = useState(null);
  const [merchantName, setMerchantName] = useState(null);
  const [showNotification, setShowNotification] = useState(false);

  const merchantMapping = {
    "Singapore Airlines": "SIA",
    "McDonald's": "MCD",
    "Zalora Apparels": "ZAL"
  };

  function amountChanged(e) {
    e.preventDefault();
    console.log(e.target.value);

    setAmount(e.target.value);

    // store locally
    localStorage.setItem("transactionAmount", e.target.value);
  }

  function handleMerchantChange(e) {
    e.preventDefault();
    console.log(e.target.innerHTML);
    setMerchantName(e.target.innerHTML);
    setMerchant(merchantMapping[e.target.innerHTML]);

    // store locally
    localStorage.setItem("merchantName", e.target.innerHTML);
    localStorage.setItem("merchant", merchantMapping[e.target.innerHTML]);
  }

  function completeTransaction() {
    setShowNotification(true);
  }

  return (
    <div>
      <div className="row">
        <img
          className="nav-arrow"
          src="/images/navigation/back.png"
          onClick={props.goBack}
        ></img>
      </div>
      <h1 className="screen-title">Transact</h1>
      <span className="instructions">
        Configure your transaction below on the amount and merchant you are
        transferring to.
      </span>
      {showNotification ? (
        <NotificationBubble
          notificationBubbleTapped={props.notificationBubbleTapped}
        />
      ) : null}
      <img className="transact-image" src="/images/app/transact.png" />

      <div className="row card-details">
        <div className="col-sm-12">
          <span className="">Funding source: XXXX-XXXX-XXXX-1281</span>
        </div>
      </div>
      <br />
      <div className="row">
        <div className="col-sm-12">
          <div className="input-group mb-3">
            <div className="input-group-prepend">
              <span className="input-group-text" id="inputGroup-sizing-default">
                Amount
              </span>
            </div>
            <input
              type="number"
              className="form-control"
              aria-label="Sizing example input"
              aria-describedby="inputGroup-sizing-default"
              onChange={amountChanged}
            />
          </div>
        </div>
      </div>
      <div className="dropdown">
        <button
          className="btn btn-secondary dropdown-toggle"
          type="button"
          id="dropdownMenuButton"
          data-toggle="dropdown"
          aria-haspopup="true"
          aria-expanded="false"
        >
          Select Merchant
        </button>
        <div
          className="dropdown-menu"
          aria-labelledby="dropdownMenuButton"
          onClick={handleMerchantChange}
        >
          <a className="dropdown-item">KFC</a>
          <a className="dropdown-item">Singapore Airlines</a>
          <a className="dropdown-item">Zalora</a>
        </div>
        {merchant !== null ? (
          <div className="merchant-info">
            You have selected <b>{merchantName}</b> for this transaction.
          </div>
        ) : null}
      </div>
      <div className="row">
        <div className="col-sm-12 text-center transction-button">
          <button
            type="button"
            className="btn btn-primary"
            onClick={completeTransaction}
          >
            Confirm transaction
          </button>
        </div>
      </div>
    </div>
  );
}

export default Transact;

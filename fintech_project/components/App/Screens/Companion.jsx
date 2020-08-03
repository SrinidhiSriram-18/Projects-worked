import React, { useState, useEffect } from "react";

function Companion(props) {
  const [transactionAmount, setTransactionAmount] = useState(0);

  function returnWarning() {
    return (
      <div>
        <br />
        Based on your spending, we recommend that you need to be more{" "}
        <b>cautious</b>.<br />
        <br /> You are currently 18 years old and if you continue{" "}
        <b>saving at least $1000 a month</b>, you will be able to enrol for
        university courses at the National University of Singapore{" "}
        <b>without taking a student loan</b>!
      </div>
    );
  }

  function returnOnTrack() {
    return (
      <div>
        You're well on track to saving some amount of money each month. Good
        job!!
      </div>
    );
  }

  useEffect(() => {
    const amount = localStorage.getItem("totalTransactionAmount");
    console.log(amount);
    if (amount !== null) {
      setTransactionAmount(parseFloat(amount).toFixed(2));
    }
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
      <h1 className="screen-title">Lifelong Companion</h1>
      <img className="companion-image" src="/images/app/friend.png" />
      <img className="timeline-image" src="/images/app/timeline.png" />

      <div className="row">
        <div className="col-sm-12">
          {transactionAmount === 0 ? (
            <div>
              <br />
              <b>Please upload some transaction data first!</b>
            </div>
          ) : transactionAmount > 10000 ? (
            <div>
              Your budget for this month is <b>$10000</b>. You have spent a
              total of <b>${transactionAmount}</b>. <br /> {returnWarning()}
            </div>
          ) : (
            returnOnTrack()
          )}
        </div>
      </div>
    </div>
  );
}

export default Companion;

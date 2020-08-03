import React, { useEffect, useState } from "react";

function Transactions(props) {
  const transactions = props.transactions;
  const transactionsCollection = transactions.map((transaction, idx) => (
    <div className="transaction-row" key={idx}>
      <div className="row">
        <div className="col-sm-12">
          <div className="transaction-amount">{transaction[2]}</div>
        </div>
      </div>
      <div className="row">
        <div className="col-sm-6">
          <div className="merchant-name">{transaction[1]}</div>
        </div>
        <div className="col-sm-6">
          <div className="transaction-date float-right">{transaction[0]}</div>
        </div>
      </div>
    </div>
  ));
  return <div>{transactionsCollection}</div>;
}

function TransactionHistory(props) {
  const [transactions, setTransactions] = useState([]);
  useEffect(() => {
    // retrieve from localStorage
    if (localStorage.getItem("transactionsList") !== null) {
      setTransactions(JSON.parse(localStorage.getItem("transactionsList")));
      console.log(transactions);
    }
  }, []);
  return (
    <div className="transaction-history">
      <div className="row">
        <img
          className="nav-arrow"
          src="/images/navigation/back.png"
          onClick={props.goBack}
        ></img>
      </div>
      <h1 className="screen-title">Transaction History</h1>
      <div className="instructions"></div>
      <img className="related-offer-image" src="/images/app/history.png" />
      <br />
      {transactions.length === 0 ? (
        <span className="instructions">
          You do not seem to have any transactions at the moment!
        </span>
      ) : (
        <div>
          <div className="instructions">Check out your transactions below.</div>
          <br />
          <Transactions transactions={transactions} />
        </div>
      )}
    </div>
  );
}

export default TransactionHistory;

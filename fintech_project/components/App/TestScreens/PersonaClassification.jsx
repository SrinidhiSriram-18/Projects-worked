import React, { useState, useCallback } from "react";
import ReactLoading from "react-loading";
import axios from "axios";
import CSVReader from "react-csv-reader";

function UploadData(props) {
  const [transactionData, setTransactionData] = useState({});
  const [isLoading, setLoading] = useState(false);

  function submitData(e) {
    e.preventDefault();
    console.log("Upload initiated");
    setLoading(true);

    // upload data
    axios
      .post("/api/upload-data", transactionData)
      .then(function(response) {
        props.updateNotification(response.data);
        // store persona in local storage
        localStorage.setItem("persona", response.data.persona);

        // store total transaction amount
        localStorage.setItem(
          "totalTransactionAmount",
          response.data.transactionAmount
        );
        setLoading(false);
      })
      .catch(function(error) {
        console.log(error);
        setLoading(false);
      });
  }

  function handleCSVUpload(data) {
    console.log(data);

    // store in localStorage
    try {
      localStorage.setItem("transactionsList", JSON.stringify(data));
      setTransactionData(data);
    } catch (e) {
      console.error(e);
    }
  }

  return (
    <div>
      <h1>Persona Classification</h1>
      <h4>About the feature</h4>
      <div className="instructions">
        A bank's asset is its data. And when you think about a bank's data,
        there are literally millions of transactions flowing through their
        servers every single second. Persona classification is a feature which
        uses transaction data of a person to classify him or her into a persona.
        This allows the bank to provide more relevant products and notifications
        for this person.
      </div>
      <br />
      <div className="instructions">
        Please upload your monthly transaction data in CSV format:
      </div>
      <div className="instructions">
        <div>Example: date,merchant,amount</div>
        <span>01/10/2019,McDonald's,20.30</span>
      </div>
      <br />
      <div className="">
        Download some sample data to start with:
        <br />
        <a href="/sample-data/sample-data-foodie.csv" download>
          Download Sample 1 - Foodie Persona
        </a>
        <br />
        <a href="/sample-data/sample-data-traveller.csv" download>
          Download Sample 2 - Traveller Persona
        </a>
        <br />
        <a href="/sample-data/sample-data-fashion.csv" download>
          Download Sample 3 - Fashion Fiesta Persona
        </a>
      </div>

      {isLoading ? (
        <div className="text-center">
          <h6>Loading...</h6>
          <ReactLoading
            type={"spinningBubbles"}
            color={"#4287f5"}
            height={64}
            width={64}
            className="loader"
          />
        </div>
      ) : null}

      <CSVReader cssClass="csv-upload" onFileLoaded={handleCSVUpload} />
      <br />
      <button type="button" className="btn btn-primary" onClick={submitData}>
        Submit
      </button>
    </div>
  );
}

export default UploadData;

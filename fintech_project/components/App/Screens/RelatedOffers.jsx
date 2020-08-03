import Reat, { useState, useEffect } from "react";
import axios from "axios";

function Offers(props) {
  const offersRetrieved = props.offers;
  console.log(offersRetrieved);
  const offerCollection = offersRetrieved.map((offer, idx) => (
    <div className="row offer" key={idx}>
      <div className="col-sm-2">
        <img className="offer-icon vertical-middle" src={offer.imageUrl} />
      </div>
      <div className="col-sm-8">
        <span className="offer-title">{offer.title}</span>
        <br />
        <span className="offer-body">{offer.message}</span>
      </div>
      <div className="col-sm-2">
        <img
          className="vertical-middle offer-icon"
          src="images/navigation/forward.png"
        />
      </div>
    </div>
  ));
  return <div>{offerCollection}</div>;
}

function RelatedOffers(props) {
  const [merchantName, setMerchantName] = useState(null);
  const [offers, setOffers] = useState([]);
  const [transactionAmount, setTransactionAmount] = useState(null);

  useEffect(() => {
    // get local items
    setMerchantName(localStorage.getItem("merchantName"));
    setTransactionAmount(localStorage.getItem("transactionAmount"));

    async function getOffers() {
      const offersList = await axios.get(
        "/api/offers?merchant=" + localStorage.getItem("merchantName")
      );
      console.log(offersList.data);
      setOffers(offersList.data);
    }
    getOffers();
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
      <h1 className="screen-title">Related Offers</h1>
      <div className="instructions">
        We saw you just made a transaction of <b>SGD {transactionAmount}</b> to{" "}
        <b>{merchantName}</b> and we feel the following offers will be useful or
        related to you!
      </div>
      <img
        className="related-offer-image"
        src="/images/app/related_offers.png"
      />
      <br />
      <Offers offers={offers} />
    </div>
  );
}

export default RelatedOffers;

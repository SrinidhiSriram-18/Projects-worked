function Notification(props) {
  return (
    <div className="container">
      <div className="row" onClick={props.notificationTapped}>
        <div className="col-sm-12 notification">
          <img className="notification-app-icon" src="/images/app/bank.png" />{" "}
          <span className="notification-title">Bank of NUS</span>
          <br />
          <span className="notification-body">{props.messageBody}</span>
        </div>
      </div>
    </div>
  );
}

export default Notification;

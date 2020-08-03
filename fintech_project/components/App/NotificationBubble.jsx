function NotificationBubble(props) {
  function handleNotificationTap() {}
  return (
    <div className="container text-center">
      <div className="row" onClick={props.notificationTapped}>
        <div className="col-sm-12 notification-bubble">
          <span className="notification-bubble-title">New offers!</span>
          <br />
          <span className="notification-bubble-body">
            Based on your transaction, we think you will find these useful!
          </span>
          <hr className="horizontal-line-bubble" />
          <span
            className="notification-action-button"
            onClick={props.notificationBubbleTapped}
          >
            Show me
          </span>
        </div>
      </div>
    </div>
  );
}

export default NotificationBubble;

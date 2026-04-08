import azure.functions as func
import json
import logging
import os
from azure.communication.callautomation import CallAutomationClient
from azure.communication.callautomation import PhoneNumberIdentifier

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="ForwardIncomingCall", auth_level=func.AuthLevel.ANONYMOUS)
def ForwardIncomingCall(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Incoming call event received")

    try:
        body = req.get_json()

        # Handle Event Grid validation handshake
        if isinstance(body, list) and body[0].get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
            logging.info("Handling Event Grid validation")
            validation_code = body[0]["data"]["validationCode"]
            return func.HttpResponse(
                json.dumps({"validationResponse": validation_code}),
                status_code=200,
                mimetype="application/json"
            )

        # Redirect incoming call to +18138510780
        event = body[0] if isinstance(body, list) else body
        incoming_call_context = event.get("data", {}).get("incomingCallContext")

        if not incoming_call_context:
            logging.error("No incomingCallContext found")
            return func.HttpResponse("Missing incomingCallContext", status_code=400)

        connection_string = os.environ["ACS_CONNECTION_STRING"]
        forward_to_number = "+18138510780"

        client = CallAutomationClient.from_connection_string(connection_string)

        client.redirect_call(
            incoming_call_context=incoming_call_context,
            target_participant=PhoneNumberIdentifier(forward_to_number)
        )

        logging.info("Call successfully redirected to +18138510780")
        return func.HttpResponse("OK", status_code=200)

    except Exception as e:
        logging.error(f"Error redirecting call: {str(e)}")
        return func.HttpResponse(str(e), status_code=500)
from __future__ import annotations

from pathlib import Path

from glaucoma.common import ARTIFACTS, digest, exclusive_output, read_json, write_json, sha256
from .results import artifact_identity, export_predictions, summary, verify_artifact


def reserve(context, artifact, milestone, output, ledger_path=None):
    if milestone not in ("candidate", "delivery"):
        raise ValueError("Plan3 preserves the already-used baseline; only candidate/delivery are available")
    verify_artifact(context, artifact, require_full=milestone == "delivery")
    ledger_path = Path(ledger_path or ARTIFACTS / "valid_milestones.json")
    signature = artifact_identity(artifact)
    with exclusive_output(ledger_path.with_suffix(".lock")):
        ledger = read_json(ledger_path)
        if "baseline" not in ledger:
            raise ValueError("Central ledger must retain the existing baseline")
        if milestone in ledger and ledger[milestone]["artifact_sha256"] != signature:
            raise ValueError("Milestone already reserved for a different frozen artifact")
        if milestone == "delivery" and ledger.get("candidate", {}).get("status") != "complete":
            raise ValueError("Complete candidate confirmation before reserving delivery")
        if milestone not in ledger:
            ledger[milestone] = {"artifact_sha256": signature, "status": "reserved", "protocol": "plan3"}
            write_json(ledger_path, ledger)
        ticket = {"format": "plan3-valid-ticket-v1", "milestone": milestone, "artifact_sha256": signature,
                  "data_fingerprint": context.audit["data_fingerprint"], "fold_fingerprint": context.audit["fold_fingerprint"],
                  "valid_rows": context.audit.get("splits", {}).get("valid", {}).get("rows"),
                  "ledger": ledger, "ledger_sha256": digest(ledger)}
        write_json(output, ticket)
    return ticket


def check_ticket(ticket, artifact, audit):
    milestone = ticket.get("milestone")
    if (ticket.get("format") != "plan3-valid-ticket-v1" or milestone not in ("candidate", "delivery") or
            "baseline" not in ticket.get("ledger", {}) or digest(ticket["ledger"]) != ticket["ledger_sha256"] or
            ticket["ledger"].get(milestone, {}).get("artifact_sha256") != ticket["artifact_sha256"] or
            artifact_identity(artifact) != ticket["artifact_sha256"] or
            any(ticket[k] != audit[k] for k in ("data_fingerprint", "fold_fingerprint"))):
        raise ValueError("Missing/stale evaluation ticket or changed frozen artifact; cannot read valid labels")


def evaluate(context, artifact, ticket_path, output, device, workers=0, batch_size=4):
    ticket, output = read_json(ticket_path), Path(output)
    check_ticket(ticket, artifact, context.audit)  # Must happen before loading official-valid rows.
    write_json(output / "evaluation_reservation.json", ticket)
    rows, probabilities = export_predictions(context, artifact, "valid", output / "valid.csv", device, workers, batch_size)
    result = {"format": "plan3-valid-result-v1", "milestone": ticket["milestone"],
              "artifact_sha256": ticket["artifact_sha256"], "ticket_sha256": digest(ticket),
              "data_fingerprint": context.audit["data_fingerprint"], "fold_fingerprint": context.audit["fold_fingerprint"],
              "report": summary(rows, probabilities), "prediction_sha256": sha256(output / "valid.csv"), "status": "complete"}
    write_json(output / "valid_report.json", result)
    return result


def accept_result(path, ledger_path=None):
    result = read_json(path)
    ticket = read_json(Path(path).parent / "evaluation_reservation.json")
    if (result.get("format") != "plan3-valid-result-v1" or result.get("status") != "complete" or
            digest(ticket) != result["ticket_sha256"] or result["artifact_sha256"] != ticket["artifact_sha256"] or
            result["milestone"] != ticket["milestone"] or
            any(result[k] != ticket[k] for k in ("data_fingerprint", "fold_fingerprint"))):
        raise ValueError("Incomplete/unbound evaluation result")
    predictions = Path(path).parent / "valid.csv"
    if sha256(predictions) != result["prediction_sha256"]:
        raise ValueError("Official-valid predictions changed after evaluation")
    from .results import read_predictions
    from glaucoma.metrics import metrics
    rows = read_predictions(predictions)
    actual = metrics([int(r["True_Label"]) for r in rows], [float(r["Probability"]) for r in rows])
    if actual != result["report"]["full"] or ticket["valid_rows"] is not None and len(rows) != ticket["valid_rows"]:
        raise ValueError("Official-valid metrics/row count do not match the saved predictions")
    ledger_path = Path(ledger_path or ARTIFACTS / "valid_milestones.json")
    with exclusive_output(ledger_path.with_suffix(".lock")):
        ledger = read_json(ledger_path)
        entry = ledger.get(result["milestone"], {})
        if entry.get("artifact_sha256") != result["artifact_sha256"]:
            raise ValueError("Remote result does not match the central reservation")
        entry.update(status="complete", metrics=result["report"]["full"], result_sha256=digest(result))
        write_json(ledger_path, ledger)
    return entry

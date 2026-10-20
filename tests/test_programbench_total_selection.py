import pytest

from agent_company.programbench.total_selection import choose_mode, compare_totals, confirm_totals


def result(passed, total=1000):
    return {"status": "failed", "score": passed / total,
            "metrics": {"verifier": {"status": "completed", "reward": passed / total,
                                     "passed_tests": passed, "valid_tests": total}}}


@pytest.mark.parametrize("a,b,expected", [(100,149,"not_distinguished"), (100,150,"needs_confirmation"),
                                           (0,19,"not_distinguished"), (0,20,"needs_confirmation"),
                                           (0,0,"not_distinguished"), (100,100,"not_distinguished")])
def test_total_threshold_both_conditions_and_no_efficiency_path(a,b,expected):
    assert compare_totals(result(a), result(b))["decision"] == expected


def test_missing_is_unknown_but_failed_agent_zero_is_valid():
    assert compare_totals(result(0), result(20))["decision"] == "needs_confirmation"
    missing = result(0)
    missing["score"] = None
    assert compare_totals(missing, result(20))["decision"] == "inconclusive"
    warning = result(20)
    warning["metrics"]["verifier"]["reward"] = None
    assert compare_totals(result(0), warning)["decision"] == "inconclusive"
    assert compare_totals(result(0), result(20, 2000))["decision"] == "inconclusive"


def test_confirm_uses_saved_trigger_and_direction_not_new_range_or_new_base():
    trigger = compare_totals(result(100), result(150))
    assert trigger["threshold_passed_tests"] == 50
    # A newly estimated threshold would be 200 here
    # The saved half is 25.
    assert confirm_totals(trigger, result(400), result(425), concurrent_pair=True)["decision"] == "confirmed"
    assert confirm_totals(trigger, result(400), result(424), concurrent_pair=True)["decision"] == "not_confirmed"
    assert confirm_totals(trigger, result(425), result(400), concurrent_pair=True)["decision"] == "not_confirmed"
    assert confirm_totals(trigger, result(400), result(425), concurrent_pair=False)["decision"] == "inconclusive"


def test_same_condition_history_updates_only_at_three_runs():
    assert compare_totals(result(100), result(150), same_condition_team_counts=[100,200])["threshold_passed_tests"] == 50
    assert compare_totals(result(100), result(150), same_condition_team_counts=[100,200,100])["threshold_passed_tests"] == 200


def test_mode_selection_defaults_and_keeps_ties_for_user():
    assert choose_mode([])["mode"] == "propose"
    trigger=compare_totals(result(100), result(150))
    confirm=confirm_totals(trigger,result(100),result(125),concurrent_pair=True)
    trigger.update(incumbent_execution_mode="single", candidate_execution_mode="team", incumbent_trial="single", candidate_trial="global", candidate_role_policy_mode="global")
    confirm["trigger_trials"]=["single","global"]
    row={"mode":"global","initial":trigger,"confirmation":confirm}
    assert choose_mode([row])["mode"] == "global"
    other={"mode":"master","initial":{**trigger,"candidate_trial":"master","candidate_role_policy_mode":"master"},
           "confirmation":{**confirm,"trigger_trials":["single","master"]}}
    assert choose_mode([row,other])["mode"] is None


@pytest.mark.parametrize("incumbent,candidate", [("team", "team"), (None, "team"), ("single", "single")])
def test_choose_mode_rejects_non_single_initial_incumbent(incumbent, candidate):
    with pytest.raises(ValueError, match="Single incumbent"):
        choose_mode([{"mode": "global", "initial": {
            "incumbent_execution_mode": incumbent, "candidate_execution_mode": candidate}}])


@pytest.mark.parametrize("lower,total,boundary", [(101, 1000, 51), (0, 1001, 21), (3, 51, 2)])
def test_integer_threshold_rounds_each_required_lower_bound_up(lower, total, boundary):
    below = compare_totals(result(lower, total), result(lower + boundary - 1, total))
    equal = compare_totals(result(lower, total), result(lower + boundary, total))
    assert below["threshold_passed_tests"] == equal["threshold_passed_tests"] == boundary
    assert below["decision"] == "not_distinguished"
    assert equal["decision"] == "needs_confirmation"
    # Odd saved thresholds require the confirmation gap to round upward too.
    half = (boundary + 1) // 2
    assert confirm_totals(equal, result(lower, total), result(lower + half - 1, total),
                          concurrent_pair=True)["decision"] == "not_confirmed"
    assert confirm_totals(equal, result(lower, total), result(lower + half, total),
                          concurrent_pair=True)["decision"] == "confirmed"


@pytest.mark.parametrize("case", ["missing_confirmation", "inconclusive_initial",
                                  "inconclusive_confirmation", "incumbent_direction"])
def test_mode_selection_does_not_promote_missing_unknown_or_single_advantage(case):
    first = compare_totals(result(100), result(150))
    first.update(incumbent_execution_mode="single", candidate_execution_mode="team",
                 incumbent_trial="single", candidate_trial="global", candidate_role_policy_mode="global")
    confirmation = confirm_totals(first, result(100), result(125), concurrent_pair=True)
    confirmation["trigger_trials"] = ["single", "global"]
    row = {"mode": "global", "initial": first, "confirmation": confirmation}
    if case == "missing_confirmation":
        row.pop("confirmation")
    elif case == "inconclusive_initial":
        first.update(decision="inconclusive", direction=None, passed_difference=None)
    elif case == "inconclusive_confirmation":
        confirmation["decision"] = "inconclusive"
    else:
        first.update(direction="incumbent", passed_difference=-50)
        confirmation["direction"] = "incumbent"
    assert choose_mode([row]) == {"mode": "propose", "reason": "user_default_no_confirmed_team_advantage"}

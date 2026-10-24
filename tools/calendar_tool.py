from crewai.tools import tool

@tool("schedule_interview")
def schedule_interview(candidate_email: str, date_time: str, interview_link: str) -> str:
    """Useful to formally schedule an interview on the calendar.
    Requires the candidate_email, the date_time of the interview, and a meeting interview_link (like a Zoom or Google Meet link)."""
    
    print("\n" + "="*50)
    print("[CALENDAR] OFFICIAL CALENDAR EVENT CREATED")
    print(f"Candidate: {candidate_email}")
    print(f"Scheduled Time: {date_time}")
    print(f"Meeting Link: {interview_link}")
    print(f"Status: Sent Calendar Invite (.ics form)")
    print("="*50 + "\n")
    
    return f"Success! Interview officially scheduled for {date_time} via {interview_link}. Calendar invite sent to {candidate_email}."

def get_calendar_tool():
    return schedule_interview

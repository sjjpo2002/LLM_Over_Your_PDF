css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://w7.pngwing.com/pngs/453/623/png-transparent-robot-free-computer-icons-robot-electronics-text-rectangle-thumbnail.png">
    </div>
    <div class="message">{{place_holder}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/106/106137.png">
    </div>    
    <div class="message">{{place_holder}}</div>
</div>
"""

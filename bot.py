import os
import telebot
from predictor import Predictor
from flask import Flask, request

TOKEN = '1008454305:AAHKHXFQYphDITpwISxvDD0W-hr35Bwt05s'
bot = telebot.TeleBot(TOKEN)
server = Flask(__name__)
model = Predictor()


@bot.message_handler(content_types=['text'])
def predict_joke(message):
    res = 'смешно' if model.predict(message.text) == 1 else 'не смешно'
    bot.send_message(message.chat.id, 'Мне было ' + str(res))


@bot.message_handler(commands=['help'])
def get_help(message):
    bot.send_message(message.chat.id, 'Попробуй меня рассмешить:)')


@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Привет, ' + message.from_user.first_name + " !")


@server.route('/' + TOKEN, methods=['POST'])
def get_message():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://jokes-checker.herokuapp.com/' + TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))

import streamlit as st
from multiapp import MultiApp
from apps import  dolar, euro #, model1 # import your app modules here

app = MultiApp()

st.markdown("""
#  Minería de Datos - Grupo 2 - Modelo LSTM

""")

st.title('Integrantes:')

st.write('- Marcos Valdez, Alexander Junior')
st.write('- Navarro Ortiz, Eduardo')
st.write('- Quinteros Peralta, Rodrigo Ervin')
st.write('- Rojas Miñan, Alexis Luis Clemente')
st.write('- Tirado Julca Juan Jose')
st.write('- Valentin Ricaldi David Frank')

st.title('Escoja la divisa:')

# Add all your application here
#app.add_app("Home", home.app)
app.add_app("Dolares", dolar.app)
app.add_app("Euros", euro.app)
#app.add_app("Euros", model1.app)
# The main app
app.run()




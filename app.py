import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title='App Success Predictor', page_icon='📱')


@st.cache_resource
def load_model():
    """Loads the saved pipeline (preprocessing + Random Forest) and the chosen threshold."""
    model = joblib.load('final_model.pkl')
    threshold = joblib.load('threshold.pkl')
    return model, threshold


model, threshold = load_model()

# Read the dropdown options directly from the trained OneHotEncoder
encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
category_options = list(encoder.categories_[0])
content_rating_options = list(encoder.categories_[1])
android_options = [1.0, 2.0, 2.2, 2.3, 3.0, 4.0, 4.1, 4.2, 4.3, 4.4, 5.0, 5.1, 6.0, 7.0, 7.1, 8.0, 9.0, 10.0, 11.0]

st.title('📱 Google Play App Success Predictor')
tab_predict, tab_insights = st.tabs(['🔮 Predictor', '📊 Insights'])

with tab_predict:
    st.write('Enter the details of an app to predict whether it will reach **500,000+ installs**.')

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox('Category', category_options, index=category_options.index('Tools'))
        content_rating = st.selectbox('Content Rating', content_rating_options, index=content_rating_options.index('Everyone'))
        price = st.number_input('Price ($)  —  0 means free', min_value=0.0, max_value=400.0, value=0.0, step=0.5)
        size_mb = st.number_input('Size (MB)', min_value=0.0, max_value=2000.0, value=20.0, step=1.0)
        min_android = st.selectbox('Minimum Android Version', android_options, index=android_options.index(4.1))
    with col2:
        released_year = st.slider('Release Year', 2010, 2021, 2018)
        last_updated_year = st.slider('Last Updated Year', 2010, 2021, 2021)
        ad_supported = st.checkbox('Has Ads')
        in_app_purchases = st.checkbox('Has In-App Purchases')

    if st.button('Predict', type='primary'):
        # Build one row with the same column names the model was trained on
        app = pd.DataFrame([{
            'Category': category,
            'Free': price == 0,
            'Price': price,
            'Content Rating': content_rating,
            'Ad Supported': ad_supported,
            'In App Purchases': in_app_purchases,
            'Size_MB': size_mb,
            'Min_Android': min_android,
            'Released_Year': released_year,
            'Last_Updated_Year': last_updated_year,
        }])
        prob = model.predict_proba(app)[0][1]

        if prob >= threshold:
            st.success(f'✅ This app is likely to be **SUCCESSFUL** (success probability: {prob:.0%})')
        else:
            st.error(f'❌ This app is **NOT** likely to be successful (success probability: {prob:.0%})')
        st.caption(f'The model says "successful" only when the probability is at least {threshold:.0%}.')

with tab_insights:
    st.write('Key findings from the analysis of **2.3 million apps** on Google Play (data collected in 2021).')

    # Numbers below were computed on the full dataset in the notebook (EDA section)
    col1, col2, col3 = st.columns(3)
    col1.metric('Apps analysed', '2.3M')
    col2.metric('Successful apps', '3.2%')
    col3.metric('Model precision / recall', '0.34 / 0.42')

    st.subheader('1. Game categories have the highest success rate')
    categories = pd.DataFrame({
        'Category': ['Simulation', 'Role Playing', 'Racing', 'Weather', 'Strategy',
                     'Action', 'Casino', 'Video Players & Editors', 'Card', 'Photography'],
        'Success Rate %': [18.2, 17.0, 16.21, 11.69, 11.48, 11.0, 10.82, 10.43, 10.0, 8.38]
    })
    st.plotly_chart(px.bar(categories, x='Category', y='Success Rate %', text_auto=True), use_container_width=True)

    st.subheader('2. In-app purchases and ads are linked to success')
    features = pd.DataFrame({
        'Feature': ['In-App Purchases', 'In-App Purchases', 'Ads', 'Ads'],
        'Has it': ['No', 'Yes', 'No', 'Yes'],
        'Success Rate %': [2.1, 15.57, 1.49, 5.01]
    })
    st.plotly_chart(px.bar(features, x='Feature', y='Success Rate %', color='Has it',
                           barmode='group', text_auto=True), use_container_width=True)

    st.subheader('3. Older apps had more time to collect downloads')
    years = pd.DataFrame({
        'Release Year': list(range(2010, 2022)),
        'Success Rate %': [27.97, 19.24, 14.89, 11.43, 9.65, 6.67, 5.32, 3.82, 2.77, 1.96, 1.23, 0.49]
    })
    st.plotly_chart(px.line(years, x='Release Year', y='Success Rate %', markers=True), use_container_width=True)

if predict_btn:
    if input_text.strip() == "":
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
    else:
        with st.spinner('Menganalisis...'):
            try:
                stopword_remover = tools['stopword']
                stemmer = tools['stemmer']
                processed = preprocess_text(input_text, stopword_remover, stemmer)
                vec = vectorizer.transform([processed])

                pred_bnb = model_bnb.predict(vec)[0]
                pred_svm = model_svm.predict(vec)[0]
                pred_ensemble = model_ensemble.predict(vec)[0]

                prob_bnb = model_bnb.predict_proba(vec)[0]
                prob_svm = model_svm.predict_proba(vec)[0]
                prob_ensemble = model_ensemble.predict_proba(vec)[0]

                st.subheader("🎯 Hasil Analisis (Ensemble)")

                max_prob = max(prob_ensemble) * 100
                conf_text, conf_type = get_confidence_badge(max_prob)

                if pred_ensemble == "positive":
                    st.success("### ✅ Sentimen: POSITIF")
                else:
                    st.error("### ❌ Sentimen: NEGATIF")

                st.info(f"**Tingkat Keyakinan:** {conf_text} ({max_prob:.1f}%)")

                # Probabilitas
                st.write("**📊 Probabilitas:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Negatif", f"{prob_ensemble[0]*100:.1f}%")

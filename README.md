# QuickRAG Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

A user-friendly end-to-end Retrieval-Augmented Generation (RAG) application that allows you to upload documents and ask questions about their content using advanced AI.

## 🌟 What is QuickRAG?

QuickRAG Demo is an interactive web application built with Streamlit that demonstrates how to create a powerful question-answering system over your documents. Simply upload PDF files, and the app will:

- **Extract and process** text from your documents
- **Create semantic embeddings** for intelligent search
- **Answer questions** about your content using AI
- **Provide source citations** so you know where answers come from

## 🚀 Features

- **Easy File Upload**: Drag-and-drop PDF support
- **Smart Document Processing**: Automatic text extraction and chunking
- **Semantic Search**: Find relevant information using vector embeddings
- **AI-Powered Q&A**: Get accurate answers with source references
- **Clean Interface**: Intuitive Streamlit UI for seamless interaction
- **Real-time Processing**: See results as they're generated

## 🛠️ Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/venkatasaiv/quickrag-demo.git
   cd quickrag-demo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` to access the app

## 📖 How to Use

### Step 1: Upload Documents
- Click "Browse files" or drag-and-drop PDF files into the upload area
- The app supports multiple PDF files simultaneously
- Wait for the "Documents processed successfully!" message

### Step 2: Ask Questions
- Type your question in the text input field
- Click "Get Answer" or press Enter
- View the AI-generated response with source citations

### Step 3: Explore Results
- Review the answer and supporting evidence
- Check the source references to verify information
- Ask follow-up questions to dive deeper

## 💡 Example Use Cases

- **Research**: Quickly find information across multiple research papers
- **Legal Documents**: Search through contracts and legal texts
- **Technical Manuals**: Get instant answers from documentation
- **Educational Content**: Study materials and textbooks
- **Business Reports**: Extract insights from financial and business documents

## 🔧 Configuration

The app can be customized by modifying these key parameters in the code:

- **Chunk Size**: Adjust document splitting granularity
- **Model Selection**: Choose different AI models for embeddings and generation
- **Search Results**: Configure number of relevant chunks to retrieve
- **UI Elements**: Customize the Streamlit interface

## 📦 Dependencies

Key libraries used in this project:

- `streamlit` - Web app framework
- `langchain` - RAG implementation framework
- `openai` - AI model integration
- `pypdf2` or `pdfplumber` - PDF processing
- `faiss-cpu` or `chromadb` - Vector database
- `python-dotenv` - Environment variable management

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)

## ❓ FAQ

**Q: What file formats are supported?**
A: Currently, the app supports PDF files. Support for additional formats (Word, TXT) can be added.

**Q: How many documents can I upload?**
A: There's no hard limit, but performance may vary based on document size and system resources.

**Q: Can I use this with other AI models?**
A: Yes! The code can be modified to work with different language models and embedding providers.

**Q: Is my data secure?**
A: Documents are processed locally and not stored permanently. API calls to AI services follow their respective privacy policies.

---

**Ready to get started?** Follow the installation steps above and start exploring your documents with AI! 🚀

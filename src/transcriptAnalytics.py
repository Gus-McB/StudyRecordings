import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

class TranscriptAnalytics:
    def __init__(self, csv_file_path):
        """
        Initialize the analytics class with a CSV transcript file.
        Expected format: start_time, end_time, speaker, text
        """
        # Try different encodings to handle various CSV formats
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                self.df = pd.read_csv(csv_file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {csv_file_path} with any encoding")
        
        self.csv_file_path = csv_file_path
        self.stop_words = set(stopwords.words('english'))
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the transcript data for analysis."""
        # Convert time columns to seconds for easier calculation
        self.df['start_seconds'] = self.df['start_time'].apply(self.time_to_seconds)
        self.df['end_seconds'] = self.df['end_time'].apply(self.time_to_seconds)
        self.df['duration'] = self.df['end_seconds'] - self.df['start_seconds']
        
        # Clean speaker names - normalize inconsistent formatting
        self.df['speaker'] = self.df['speaker'].astype(str).str.strip()  # Remove leading/trailing spaces
        self.df['speaker'] = self.df['speaker'].str.replace(r'\s+', ' ', regex=True)  # Normalize multiple spaces
        
        # Calculate pauses between utterances
        self.df = self.df.sort_values(['start_seconds']).reset_index(drop=True)
        self.df['pause_before'] = 0.0
        
        for i in range(1, len(self.df)):
            prev_end = self.df.iloc[i-1]['end_seconds']
            curr_start = self.df.iloc[i]['start_seconds']
            self.df.iloc[i, self.df.columns.get_loc('pause_before')] = max(0, curr_start - prev_end)
        
        # Clean text
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        self.df['word_count'] = self.df['clean_text'].apply(lambda x: len(x.split()))
        self.df['char_count'] = self.df['text'].apply(len)
        
        # Calculate speaking rate (words per minute)
        self.df['speaking_rate_wpm'] = (self.df['word_count'] / self.df['duration']) * 60
        self.df['speaking_rate_wpm'] = self.df['speaking_rate_wpm'].replace([np.inf, -np.inf], np.nan)
        
    def time_to_seconds(self, time_str):
        """Convert time string (HH:MM:SS or MM:SS) to seconds."""
        try:
            if isinstance(time_str, str):
                parts = time_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:  # MM:SS
                    return int(parts[0]) * 60 + int(parts[1])
            return float(time_str)
        except:
            return 0
    
    def clean_text(self, text):
        """Clean text for analysis."""
        if pd.isna(text):
            return ""
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def basic_statistics(self):
        """Generate basic speaking statistics."""
        stats = {}
        
        # Overall statistics
        total_duration = self.df['duration'].sum()
        total_words = self.df['word_count'].sum()
        
        stats['overall'] = {
            'total_speaking_time': f"{total_duration:.2f} seconds ({total_duration/60:.2f} minutes)",
            'total_words': total_words,
            'average_speaking_rate': f"{(total_words / total_duration * 60):.2f} words per minute",
            'total_utterances': len(self.df),
            'average_utterance_duration': f"{self.df['duration'].mean():.2f} seconds",
            'average_pause_duration': f"{self.df['pause_before'].mean():.2f} seconds"
        }
        
        # Per-speaker statistics
        speaker_stats = {}
        for speaker in self.df['speaker'].unique():
            speaker_data = self.df[self.df['speaker'] == speaker]
            
            speaker_duration = speaker_data['duration'].sum()
            speaker_words = speaker_data['word_count'].sum()
            
            speaker_stats[speaker] = {
                'speaking_time': f"{speaker_duration:.2f} seconds ({speaker_duration/60:.2f} minutes)",
                'percentage_of_conversation': f"{(speaker_duration/total_duration*100):.1f}%",
                'word_count': speaker_words,
                'average_speaking_rate': f"{speaker_data['speaking_rate_wpm'].mean():.2f} wpm",
                'utterance_count': len(speaker_data),
                'average_utterance_length': f"{speaker_data['duration'].mean():.2f} seconds",
                'average_pause_before_speaking': f"{speaker_data['pause_before'].mean():.2f} seconds"
            }
        
        stats['by_speaker'] = speaker_stats
        return stats
    
    def communication_effectiveness_analysis(self):
        """Analyze communication effectiveness metrics."""
        effectiveness = {}
        
        for speaker in self.df['speaker'].unique():
            speaker_data = self.df[self.df['speaker'] == speaker]
            
            # Calculate various effectiveness metrics
            avg_response_time = speaker_data['pause_before'].mean()
            speaking_consistency = 1 / (speaker_data['speaking_rate_wpm'].std() + 1)  # Lower std = more consistent
            utterance_completeness = speaker_data['duration'].mean()  # Longer utterances might indicate more complete thoughts
            
            # Sentiment analysis
            sentiments = [TextBlob(text).sentiment.polarity for text in speaker_data['text'] if pd.notna(text)]
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Vocabulary richness (unique words / total words)
            all_words = ' '.join(speaker_data['clean_text'].dropna())
            unique_words = len(set(all_words.split()))
            total_words = len(all_words.split())
            vocab_richness = unique_words / total_words if total_words > 0 else 0
            
            effectiveness[speaker] = {
                'response_time_score': max(0, 10 - avg_response_time),  # Lower response time = higher score
                'consistency_score': speaking_consistency * 10,
                'completeness_score': min(10, utterance_completeness),
                'sentiment_score': (avg_sentiment + 1) * 5,  # Convert -1,1 to 0,10 scale
                'vocabulary_richness': vocab_richness,
                'overall_effectiveness': 0  # Will be calculated
            }
            
            # Calculate overall effectiveness (weighted average)
            weights = {'response_time': 0.2, 'consistency': 0.2, 'completeness': 0.2, 'sentiment': 0.2, 'vocabulary': 0.2}
            overall = (effectiveness[speaker]['response_time_score'] * weights['response_time'] +
                      effectiveness[speaker]['consistency_score'] * weights['consistency'] +
                      effectiveness[speaker]['completeness_score'] * weights['completeness'] +
                      effectiveness[speaker]['sentiment_score'] * weights['sentiment'] +
                      effectiveness[speaker]['vocabulary_richness'] * 10 * weights['vocabulary'])
            
            effectiveness[speaker]['overall_effectiveness'] = overall
        
        return effectiveness
    
    def topic_modeling(self, n_topics=5):
        """Perform topic modeling using LDA, optimized for problem-solving conversations."""
        # Combine all text for each speaker
        speaker_texts = self.df.groupby('speaker')['clean_text'].apply(lambda x: ' '.join(x)).to_dict()
        
        if len(speaker_texts) < 2:
            return None
        
        # Prepare documents
        documents = list(speaker_texts.values())
        speakers = list(speaker_texts.keys())
        
        # Enhanced stop words for problem-solving conversations
        problem_solving_stop_words = [
            # Common conversational fillers
            'okay', 'ok', 'yeah', 'yes', 'no', 'um', 'uh', 'like', 'you', 'know', 'i', 'me', 'my',
            'we', 'us', 'our', 'it', 'its', 'that', 'this', 'the', 'a', 'an', 'and', 'or', 'but',
            'so', 'well', 'now', 'then', 'here', 'there', 'what', 'how', 'when', 'where', 'why',
            'can', 'could', 'would', 'should', 'will', 'going', 'get', 'got', 'have', 'has', 'had',
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'think', 'thought',
            'see', 'look', 'looking', 'want', 'need', 'try', 'trying', 'make', 'making', 'take', 'taking',
            'put', 'putting', 'go', 'goes', 'went', 'come', 'comes', 'came', 'say', 'said', 'tell', 'told',
            # Generic agreement/disagreement
            'alright', 'right', 'sure', 'fine', 'good', 'great', 'nice', 'cool', 'awesome',
            # Common task words that aren't specific
            'thing', 'stuff', 'something', 'anything', 'everything', 'nothing', 'way', 'ways'
        ]
        
        # Create custom stop words combining English + problem-solving specific
        custom_stop_words = list(set(problem_solving_stop_words + list(stopwords.words('english'))))
        
        # Enhanced vectorizer for problem-solving content
        vectorizer = TfidfVectorizer(
            max_features=150,  # Increased for more diverse vocabulary
            stop_words=custom_stop_words,
            ngram_range=(1, 3),  # Include trigrams for technical phrases
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8,  # Ignore words that appear in >80% of documents
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Include alphanumeric tokens
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(documents)
        except ValueError as e:
            print(f"Vectorization failed: {e}")
            return None
        
        # LDA with better parameters for technical discussions
        lda = LatentDirichletAllocation(
            n_components=min(n_topics, len(documents)), 
            random_state=42,
            max_iter=20,  # More iterations for better convergence
            learning_method='online',
            learning_offset=50,
            doc_topic_prior=0.1,  # Encourage documents to have fewer topics
            topic_word_prior=0.01  # Encourage topics to have fewer words
        )
        
        try:
            lda.fit(doc_term_matrix)
        except Exception as e:
            print(f"LDA fitting failed: {e}")
            return None
        
        # Get topics with better filtering
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        # Define topic categories for better interpretation
        topic_categories = {
            'spatial': ['left', 'right', 'up', 'down', 'top', 'bottom', 'side', 'corner', 'edge', 'center', 'middle', 'front', 'back', 'position', 'place', 'move', 'rotate', 'turn', 'flip'],
            'building': ['piece', 'block', 'connect', 'attach', 'join', 'fit', 'build', 'construct', 'assemble', 'structure', 'tower', 'wall', 'base', 'foundation'],
            'ar_vr': ['virtual', 'reality', 'ar', 'vr', 'headset', 'controller', 'interface', 'screen', 'display', 'menu', 'button', 'select', 'grab', 'hold'],
            'problem_solving': ['problem', 'solution', 'solve', 'figure', 'work', 'fix', 'adjust', 'change', 'modify', 'test', 'try', 'attempt', 'approach', 'method'],
            'collaboration': ['help', 'assist', 'together', 'team', 'share', 'communicate', 'discuss', 'explain', 'show', 'demonstrate', 'guide', 'direct']
        }
        
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words for this topic
            top_words_idx = topic.argsort()[-15:][::-1]  # Get more words for better context
            top_words = [feature_names[i] for i in top_words_idx]
            
            # Filter out remaining generic words and rank by relevance
            filtered_words = []
            for word in top_words:
                # Skip very short words and remaining generic terms
                if len(word) > 2 and word not in ['got', 'get', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
                    filtered_words.append(word)
            
            # Take top 8 meaningful words
            topics[f'Topic {topic_idx + 1}'] = filtered_words[:8]
            
            # Try to categorize the topic based on dominant themes
            category_scores = {}
            for category, keywords in topic_categories.items():
                score = sum(1 for word in filtered_words[:8] if any(keyword in word.lower() for keyword in keywords))
                if score > 0:
                    category_scores[category] = score
            
            if category_scores:
                dominant_category = max(category_scores, key=category_scores.get)
                topics[f'Topic {topic_idx + 1}'] = {
                    'words': filtered_words[:8],
                    'category': dominant_category,
                    'category_strength': category_scores[dominant_category]
                }
            else:
                topics[f'Topic {topic_idx + 1}'] = {
                    'words': filtered_words[:8],
                    'category': 'general_discussion',
                    'category_strength': 0
                }
        
        # Get topic distribution for each speaker
        topic_distributions = lda.transform(doc_term_matrix)
        speaker_topic_dist = {}
        for i, speaker in enumerate(speakers):
            speaker_topic_dist[speaker] = {
                f'Topic {j+1}': f"{prob:.3f}" for j, prob in enumerate(topic_distributions[i])
            }
        
        return {
            'topics': topics,
            'speaker_topic_distributions': speaker_topic_dist,
            'vocabulary_size': len(feature_names),
            'total_documents': len(documents)
        }
    
    def speaking_pattern_clustering(self):
        """Cluster speakers based on speaking patterns."""
        # Extract features for clustering
        features = []
        speakers = []
        
        for speaker in self.df['speaker'].unique():
            speaker_data = self.df[self.df['speaker'] == speaker]
            
            # Feature vector
            feature_vector = [
                speaker_data['duration'].mean(),  # Average utterance duration
                speaker_data['speaking_rate_wpm'].mean(),  # Average speaking rate
                speaker_data['pause_before'].mean(),  # Average pause before speaking
                speaker_data['word_count'].mean(),  # Average words per utterance
                len(speaker_data),  # Number of utterances
                speaker_data['duration'].std(),  # Variability in duration
            ]
            
            # Handle NaN values
            feature_vector = [0 if pd.isna(x) else x for x in feature_vector]
            features.append(feature_vector)
            speakers.append(speaker)
        
        if len(features) < 2:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Clustering
        n_clusters = min(3, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Results
        clustering_results = {}
        for i, speaker in enumerate(speakers):
            clustering_results[speaker] = {
                'cluster': int(clusters[i]),
                'features': {
                    'avg_duration': features[i][0],
                    'avg_speaking_rate': features[i][1],
                    'avg_pause': features[i][2],
                    'avg_words': features[i][3],
                    'utterance_count': features[i][4],
                    'duration_variability': features[i][5]
                }
            }
        
        return clustering_results
    
    def interruption_analysis(self):
        """Analyze interruption patterns."""
        interruptions = []
        
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            # Check if someone starts speaking before the previous person finishes
            # (considering a small buffer for natural conversation flow)
            if (current['start_seconds'] < previous['end_seconds'] - 0.5 and 
                current['speaker'] != previous['speaker']):
                
                interruptions.append({
                    'interrupter': current['speaker'],
                    'interrupted': previous['speaker'],
                    'time': current['start_seconds'],
                    'overlap_duration': previous['end_seconds'] - current['start_seconds']
                })
        
        # Analysis
        interruption_stats = {}
        for speaker in self.df['speaker'].unique():
            interruptions_made = len([i for i in interruptions if i['interrupter'] == speaker])
            interruptions_received = len([i for i in interruptions if i['interrupted'] == speaker])
            
            interruption_stats[speaker] = {
                'interruptions_made': interruptions_made,
                'interruptions_received': interruptions_received,
                'interruption_ratio': interruptions_made / (interruptions_received + 1)  # +1 to avoid division by zero
            }
        
        return {
            'interruption_events': interruptions,
            'interruption_stats': interruption_stats
        }
    
    def linguistic_analysis(self):
        """Perform detailed linguistic analysis."""
        linguistic_features = {}
        
        for speaker in self.df['speaker'].unique():
            speaker_data = self.df[self.df['speaker'] == speaker]
            all_text = ' '.join(speaker_data['text'].dropna())
            
            if not all_text.strip():
                continue
            
            # Sentiment analysis
            blob = TextBlob(all_text)
            sentiment = blob.sentiment
            
            # POS tagging
            words = word_tokenize(all_text.lower())
            pos_tags = pos_tag(words)
            pos_counts = {}
            for word, pos in pos_tags:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            # Calculate linguistic complexity
            sentences = sent_tokenize(all_text)
            avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0
            
            # Vocabulary diversity (Type-Token Ratio)
            unique_words = len(set(words))
            total_words = len(words)
            ttr = unique_words / total_words if total_words > 0 else 0
            
            linguistic_features[speaker] = {
                'sentiment_polarity': sentiment.polarity,
                'sentiment_subjectivity': sentiment.subjectivity,
                'average_sentence_length': avg_sentence_length,
                'type_token_ratio': ttr,
                'most_common_pos': max(pos_counts, key=pos_counts.get) if pos_counts else 'N/A',
                'pos_distribution': pos_counts
            }
        
        return linguistic_features
    
    def create_visualizations(self, output_dir='visualizations'):
        """Create various visualizations of the analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Speaking time distribution
        plt.figure(figsize=(12, 6))
        speaker_time = self.df.groupby('speaker')['duration'].sum()
        plt.subplot(2, 2, 1)
        speaker_time.plot(kind='bar')
        plt.title('Total Speaking Time by Speaker')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        # 2. Speaking rate distribution
        plt.subplot(2, 2, 2)
        for speaker in self.df['speaker'].unique():
            speaker_data = self.df[self.df['speaker'] == speaker]
            plt.hist(speaker_data['speaking_rate_wpm'].dropna(), alpha=0.7, label=speaker)
        plt.xlabel('Speaking Rate (WPM)')
        plt.ylabel('Frequency')
        plt.title('Speaking Rate Distribution')
        plt.legend()
        
        # 3. Timeline visualization
        plt.subplot(2, 2, 3)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.df['speaker'].unique())))
        speaker_colors = dict(zip(self.df['speaker'].unique(), colors))
        
        for idx, row in self.df.iterrows():
            plt.barh(row['speaker'], row['duration'], 
                    left=row['start_seconds'], 
                    color=speaker_colors[row['speaker']], alpha=0.7)
        
        plt.xlabel('Time (seconds)')
        plt.title('Speaking Timeline')
        
        # 4. Pause analysis
        plt.subplot(2, 2, 4)
        pause_data = self.df.groupby('speaker')['pause_before'].mean()
        pause_data.plot(kind='bar')
        plt.title('Average Pause Before Speaking')
        plt.ylabel('Pause Duration (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, output_file='transcript_analysis_report.txt'):
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TRANSCRIPT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis of: {self.csv_file_path}")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic Statistics
        stats = self.basic_statistics()
        report.append("1. BASIC STATISTICS")
        report.append("-" * 40)
        for key, value in stats['overall'].items():
            report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        report.append("Per-Speaker Statistics:")
        for speaker, speaker_stats in stats['by_speaker'].items():
            report.append(f"\n{speaker}:")
            for key, value in speaker_stats.items():
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Communication Effectiveness
        effectiveness = self.communication_effectiveness_analysis()
        report.append("2. COMMUNICATION EFFECTIVENESS ANALYSIS")
        report.append("-" * 40)
        for speaker, metrics in effectiveness.items():
            report.append(f"\n{speaker}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    report.append(f"  {key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Topic Modeling
        topics = self.topic_modeling()
        if topics:
            report.append("3. TOPIC MODELING")
            report.append("-" * 40)
            report.append("Identified Topics:")
            for topic_name, topic_data in topics['topics'].items():
                if isinstance(topic_data, dict):
                    words = topic_data['words']
                    category = topic_data['category']
                    report.append(f"{topic_name} ({category.replace('_', ' ').title()}): {', '.join(words)}")
                else:
                    report.append(f"{topic_name}: {', '.join(topic_data)}")
            
            report.append(f"\nVocabulary Analysis:")
            report.append(f"Total unique terms analyzed: {topics['vocabulary_size']}")
            report.append(f"Documents processed: {topics['total_documents']}")
            
            report.append("\nSpeaker-Topic Distributions:")
            for speaker, dist in topics['speaker_topic_distributions'].items():
                report.append(f"{speaker}: {dist}")
            report.append("")
        
        # Interruption Analysis
        interruptions = self.interruption_analysis()
        report.append("4. INTERRUPTION ANALYSIS")
        report.append("-" * 40)
        for speaker, stats in interruptions['interruption_stats'].items():
            report.append(f"{speaker}:")
            report.append(f"  Interruptions Made: {stats['interruptions_made']}")
            report.append(f"  Interruptions Received: {stats['interruptions_received']}")
            report.append(f"  Interruption Ratio: {stats['interruption_ratio']:.2f}")
        report.append("")
        
        # Linguistic Analysis
        linguistic = self.linguistic_analysis()
        report.append("5. LINGUISTIC ANALYSIS")
        report.append("-" * 40)
        for speaker, features in linguistic.items():
            report.append(f"\n{speaker}:")
            report.append(f"  Sentiment Polarity: {features['sentiment_polarity']:.3f}")
            report.append(f"  Sentiment Subjectivity: {features['sentiment_subjectivity']:.3f}")
            report.append(f"  Average Sentence Length: {features['average_sentence_length']:.2f} words")
            report.append(f"  Vocabulary Diversity (TTR): {features['type_token_ratio']:.3f}")
        report.append("")
        
        # Speaking Pattern Clustering
        clustering = self.speaking_pattern_clustering()
        if clustering:
            report.append("6. SPEAKING PATTERN CLUSTERING")
            report.append("-" * 40)
            for speaker, cluster_info in clustering.items():
                report.append(f"{speaker}: Cluster {cluster_info['cluster']}")
            report.append("")
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

# Example usage and batch processing
def analyse_multiple_transcripts(transcript_directory, output_directory='analysis_results'):
    """
    Analyze multiple transcript files in a directory and its subdirectories.
    Creates individual folders for each transcript file containing report, visualizations, and word clouds.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(transcript_directory):
        for filename in files:
            if filename.endswith('_transcript.csv'):
                print(f"Analyzing {filename}...")
                
                # Get the full path to the transcript file
                file_path = os.path.join(root, filename)
                base_name = filename.replace('_transcript.csv', '')
                
                # Calculate relative path from transcript_directory to current file
                rel_path = os.path.relpath(root, transcript_directory)
                
                # Create individual folder for this transcript file
                if rel_path == '.':  # Root directory
                    file_output_dir = os.path.join(output_directory, base_name)
                else:
                    file_output_dir = os.path.join(output_directory, rel_path, base_name)
                
                os.makedirs(file_output_dir, exist_ok=True)
                
                try:
                    analyser = TranscriptAnalytics(file_path)
                    
                    # Generate comprehensive report in the file's folder
                    report_file = os.path.join(file_output_dir, f'{base_name}_analysis_report.txt')
                    analyser.generate_comprehensive_report(report_file)
                    
                    # Create visualizations subdirectory within the file's folder
                    viz_dir = os.path.join(file_output_dir, 'visualizations')
                    analyser.create_visualizations(viz_dir)
                    
                    print(f"Analysis complete for {filename}. Results saved in {file_output_dir}")
                    
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")

if __name__ == "__main__":
    # Example usage - modify paths as needed
    # Single file analysis
    # analyzer = TranscriptAnalytics('path/to/your/transcript.csv')
    # print(analyzer.generate_comprehensive_report())
    
    # Batch analysis
    # analyze_multiple_transcripts('path/to/transcript/directory')
    
    print("TranscriptAnalytics module loaded. Use the class to analyse your transcript files.")
    analyse_multiple_transcripts('YOUR FILE PATH HERE')
    
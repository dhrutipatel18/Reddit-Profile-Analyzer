#!/usr/bin/env python3
"""
Reddit User Persona Generator using Together.ai API
Analyzes Reddit profiles to generate comprehensive user personas
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import praw
import requests
import streamlit as st
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_BASE_URL = "https://api.together.xyz/v1/chat/completions"

# Available models on Together.ai
AVAILABLE_MODELS = {
    "meta-llama/Llama-3-8b-chat-hf": "Llama 3 8B (Fast)",
    "meta-llama/Llama-3-70b-chat-hf": "Llama 3 70B (Best Quality)",
    "meta-llama/Llama-2-70b-chat-hf": "Llama 2 70B (Stable)",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B (Balanced)",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7B (Efficient)"
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedditDataCollector:
    """Handles Reddit data collection and processing"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._reddit_instance = None
        self._initialize_reddit_client()
    
    def _initialize_reddit_client(self):
        """Initialize Reddit client with error handling"""
        try:
            self._reddit_instance = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent="PersonaAnalyzer/2.0 (by /u/AutoPersona)"
            )
            # Test connection
            self._reddit_instance.read_only = True
            logger.info("Reddit API connection established")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def parse_reddit_url(self, url: str) -> str:
        """Extract username from various Reddit URL formats"""
        url_patterns = [
            r'reddit\.com/u/([^/?]+)',
            r'reddit\.com/user/([^/?]+)',
            r'reddit\.com/users/([^/?]+)',
            r'old\.reddit\.com/u/([^/?]+)',
            r'old\.reddit\.com/user/([^/?]+)',
            r'www\.reddit\.com/u/([^/?]+)',
            r'www\.reddit\.com/user/([^/?]+)'
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern matches, assume it's just a username
        if url and not url.startswith('http'):
            return url
        
        raise ValueError(f"Could not extract username from URL: {url}")
    
    def collect_user_content(self, username: str, max_items: int = 100) -> Dict[str, Any]:
        """Collect posts and comments from a Reddit user"""
        logger.info(f"Collecting data for user: {username}")
        
        try:
            redditor = self._reddit_instance.redditor(username)
            
            # Verify user exists
            try:
                user_id = redditor.id
                logger.info(f"User {username} found with ID: {user_id}")
            except Exception:
                raise ValueError(f"User '{username}' not found, suspended, or private")
            
            user_posts = []
            user_comments = []
            
            # Collect posts
            try:
                logger.info("Collecting user posts...")
                for idx, post in enumerate(redditor.submissions.new(limit=max_items)):
                    if idx >= max_items:
                        break
                    
                    post_data = {
                        'content_id': post.id,
                        'type': 'post',
                        'title': post.title,
                        'text': post.selftext,
                        'subreddit': str(post.subreddit),
                        'score': post.score,
                        'timestamp': post.created_utc,
                        'permalink': f"https://reddit.com{post.permalink}",
                        'is_nsfw': post.over_18
                    }
                    user_posts.append(post_data)
                
                logger.info(f"Collected {len(user_posts)} posts")
            except Exception as e:
                logger.warning(f"Error collecting posts: {e}")
            
            # Collect comments
            try:
                logger.info("Collecting user comments...")
                for idx, comment in enumerate(redditor.comments.new(limit=max_items)):
                    if idx >= max_items:
                        break
                    
                    comment_data = {
                        'content_id': comment.id,
                        'type': 'comment',
                        'text': comment.body,
                        'subreddit': str(comment.subreddit),
                        'score': comment.score,
                        'timestamp': comment.created_utc,
                        'permalink': f"https://reddit.com{comment.permalink}",
                        'parent_id': comment.parent_id if hasattr(comment, 'parent_id') else None
                    }
                    user_comments.append(comment_data)
                
                logger.info(f"Collected {len(user_comments)} comments")
            except Exception as e:
                logger.warning(f"Error collecting comments: {e}")
            
            return {
                'username': username,
                'posts': user_posts,
                'comments': user_comments,
                'collection_stats': {
                    'total_posts': len(user_posts),
                    'total_comments': len(user_comments),
                    'total_content': len(user_posts) + len(user_comments),
                    'collected_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting user data: {e}")
            raise


class TogetherAIClient:
    """Client for Together.ai API interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = TOGETHER_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def test_connection(self, model: str) -> bool:
        """Test API connection with a simple request"""
        try:
            response = self.generate_completion(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return response is not None
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def generate_completion(self, model: str, messages: List[Dict], max_tokens: int = 4000, temperature: float = 0.3) -> Optional[str]:
        """Generate completion using Together.ai API"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = self.session.post(self.base_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Unexpected API response structure: {result}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response content: {e.response.text}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return None


class PersonaAnalyzer:
    """Analyzes Reddit data to generate user personas"""
    
    def __init__(self, api_key: str):
        self.ai_client = TogetherAIClient(api_key)
        self.default_model = "meta-llama/Llama-3-70b-chat-hf"
    
    def create_persona_profile(self, user_data: Dict, model: str = None) -> str:
        """Generate comprehensive persona from user data"""
        if model is None:
            model = self.default_model
        
        logger.info(f"Generating persona using model: {model}")
        
        # Prepare content for analysis
        analysis_content = self._prepare_analysis_content(user_data)
        
        # Create analysis prompt
        analysis_prompt = self._build_analysis_prompt(user_data, analysis_content)
        
        # Generate persona
        messages = [
            {
                "role": "system",
                "content": "You are a professional digital behavioral analyst specializing in creating detailed user personas from social media activity. You analyze communication patterns, interests, and behavioral indicators to build comprehensive psychological profiles."
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]
        
        try:
            persona = self.ai_client.generate_completion(
                model=model,
                messages=messages,
                max_tokens=4000,
                temperature=0.3
            )
            
            if persona:
                logger.info("Persona generated successfully")
                return persona
            else:
                raise Exception("Failed to generate persona - empty response")
                
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            raise
    
    def _prepare_analysis_content(self, user_data: Dict) -> str:
        """Prepare user content for analysis"""
        content_samples = []
        
        # Process posts (limit to avoid token limits)
        for post in user_data['posts'][:40]:
            if post.get('title') and post.get('text'):
                content_samples.append(f"POST [{post['content_id']}] in r/{post['subreddit']}: {post['title']} | {post['text'][:400]}")
            elif post.get('title'):
                content_samples.append(f"POST [{post['content_id']}] in r/{post['subreddit']}: {post['title']}")
        
        # Process comments (limit to avoid token limits)
        for comment in user_data['comments'][:40]:
            if comment.get('text') and len(comment['text']) > 15:
                content_samples.append(f"COMMENT [{comment['content_id']}] in r/{comment['subreddit']}: {comment['text'][:400]}")
        
        return "\n\n".join(content_samples)
    
    def _build_analysis_prompt(self, user_data: Dict, content: str) -> str:
        """Build comprehensive analysis prompt"""
        stats = user_data['collection_stats']
        
        return f"""
        Analyze this Reddit user's digital behavior and create a comprehensive persona profile.
        
        USER PROFILE:
        Username: {user_data['username']}
        Total Posts: {stats['total_posts']}
        Total Comments: {stats['total_comments']}
        Total Content Analyzed: {stats['total_content']}
        
        CONTENT TO ANALYZE:
        {content}
        
        Create a detailed persona following this exact structure:
        
        🎭 REDDIT USER PERSONA: {user_data['username']}
        ================================================
        
        🎯 CORE INTERESTS & HOBBIES:
        • [Interest 1] (Evidence: "[exact quote]" - ID: [content_id])
        • [Interest 2] (Evidence: "[exact quote]" - ID: [content_id])
        • [Continue pattern...]
        
        🧠 PERSONALITY CHARACTERISTICS:
        • [Trait 1] (Evidence: "[exact quote]" - ID: [content_id])
        • [Trait 2] (Evidence: "[exact quote]" - ID: [content_id])
        • [Continue pattern...]
        
        💬 COMMUNICATION STYLE:
        • [Style element 1] (Evidence: "[exact quote]" - ID: [content_id])
        • [Style element 2] (Evidence: "[exact quote]" - ID: [content_id])
        • [Continue pattern...]
        
        💭 VALUES & WORLDVIEW:
        • [Value/Belief 1] (Evidence: "[exact quote]" - ID: [content_id])
        • [Value/Belief 2] (Evidence: "[exact quote]" - ID: [content_id])
        • [Continue pattern...]
        
        📱 REDDIT BEHAVIOR PATTERNS:
        • [Behavior 1] (Evidence: "[exact quote]" - ID: [content_id])
        • [Behavior 2] (Evidence: "[exact quote]" - ID: [content_id])
        • [Continue pattern...]
        
        🎪 COMMUNITY ENGAGEMENT:
        • [Engagement pattern 1] (Evidence: "[exact quote]" - ID: [content_id])
        • [Engagement pattern 2] (Evidence: "[exact quote]" - ID: [content_id])
        • [Continue pattern...]
        
        ANALYSIS REQUIREMENTS:
        - Use EXACT quotes from the provided content
        - Include specific content IDs for each evidence point
        - Keep quotes meaningful but concise (under 50 words)
        - Provide concrete evidence for every claim
        - Be analytical and objective
        - Focus on observable patterns and behaviors
        """


class PersonaFileHandler:
    """Handles file operations for persona storage"""
    
    def __init__(self, output_directory: str = "personas"):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_persona(self, persona_content: str, username: str, model_used: str = None) -> str:
        """Save persona to file with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{username}_persona_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        # Add metadata header
        metadata = f"""
=== REDDIT USER PERSONA ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Username: {username}
Model Used: {model_used or 'Unknown'}
Generator: Together.ai Persona Analyzer
================================

"""
        
        full_content = metadata + persona_content
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            logger.info(f"Persona saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving persona: {e}")
            raise
    
    def load_persona(self, filepath: str) -> str:
        """Load persona from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading persona: {e}")
            raise


def create_streamlit_interface():
    """Create Streamlit web interface"""
    st.set_page_config(
        page_title="Reddit Persona Analyzer",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Reddit User Persona Analyzer")
    st.markdown("*Powered by Together.ai & Advanced Language Models*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🚀 Configuration")
        
        # Model selection
        st.subheader("🤖 AI Model")
        selected_model = st.selectbox(
            "Choose Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            index=1  # Default to Llama 3 70B
        )
        
        # Analysis settings
        st.subheader("⚙️ Analysis Settings")
        content_limit = st.slider(
            "Content Limit",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
            help="Maximum posts/comments to analyze"
        )
        
        # API Status
        st.subheader("🔗 API Status")
        if st.button("🔍 Test API Connection"):
            with st.spinner("Testing connection..."):
                ai_client = TogetherAIClient(TOGETHER_API_KEY)
                if ai_client.test_connection(selected_model):
                    st.success("✅ Together.ai API: Connected")
                else:
                    st.error("❌ Together.ai API: Connection Failed")
        
        st.info("💡 **Tip**: Llama 3 70B provides the most detailed analysis")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Reddit Profile Input")
        
        # URL input
        reddit_input = st.text_input(
            "Reddit Profile URL or Username",
            placeholder="https://reddit.com/user/example OR just 'example'",
            help="Enter either a full Reddit URL or just the username"
        )
        
        # Generate button
        if st.button("🚀 Generate Persona", type="primary", use_container_width=True):
            if not reddit_input.strip():
                st.error("❌ Please provide a Reddit profile URL or username")
                return
            
            try:
                # Progress tracking
                progress = st.progress(0)
                status = st.empty()
                
                # Initialize components
                status.info("🔄 Initializing Reddit data collector...")
                progress.progress(10)
                
                collector = RedditDataCollector(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
                analyzer = PersonaAnalyzer(TOGETHER_API_KEY)
                file_handler = PersonaFileHandler()
                
                # Extract username
                status.info("👤 Processing Reddit profile...")
                progress.progress(20)
                
                username = collector.parse_reddit_url(reddit_input)
                st.info(f"📍 Analyzing user: **{username}**")
                
                # Collect data
                status.info("📊 Collecting Reddit data...")
                progress.progress(40)
                
                user_data = collector.collect_user_content(username, content_limit)
                
                # Display collection stats
                stats = user_data['collection_stats']
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Posts Found", stats['total_posts'])
                with col_stat2:
                    st.metric("Comments Found", stats['total_comments'])
                with col_stat3:
                    st.metric("Total Content", stats['total_content'])
                
                # Generate persona
                status.info(f"🤖 Generating persona with {AVAILABLE_MODELS[selected_model]}...")
                progress.progress(70)
                
                persona = analyzer.create_persona_profile(user_data, selected_model)
                
                # Save persona
                status.info("💾 Saving persona...")
                progress.progress(90)
                
                filepath = file_handler.save_persona(persona, username, AVAILABLE_MODELS[selected_model])
                
                # Complete
                progress.progress(100)
                status.success("✅ Persona generation complete!")
                
                # Display results
                st.success(f"🎉 Successfully analyzed u/{username}!")
                st.info(f"📁 Saved to: `{filepath}`")
                
                # Show persona
                st.subheader("📄 Generated Persona")
                st.text_area(
                    "Persona Content",
                    value=persona,
                    height=500,
                    disabled=True
                )
                
                # Download button
                st.download_button(
                    label="⬇️ Download Persona File",
                    data=persona,
                    file_name=f"{username}_persona.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Streamlit error: {e}")
    
    with col2:
        st.subheader("📚 Guide")
        st.markdown("""
        **How to Use:**
        1. Enter Reddit profile URL or username
        2. Select AI model and settings
        3. Click "Generate Persona"
        4. Review and download results
        
        **Input Formats:**
        - `https://reddit.com/user/example`
        - `https://reddit.com/u/example`
        - `example` (username only)
        
        **Best Practices:**
        - Use active accounts with content
        - Higher content limit = better analysis
        - Llama 3 70B for detailed insights
        - Try different models for variety
        """)
        
        st.subheader("🌟 About Together.ai")
        st.markdown("""
        **Features:**
        - Free daily credits
        - Multiple open-source models
        - Fast inference
        - High-quality outputs
        
        **Available Models:**
        - Llama 3 70B (Best quality)
        - Llama 3 8B (Fast)
        - Mixtral 8x7B (Balanced)
        - Mistral 7B (Efficient)
        """)


def create_cli_interface():
    """Create command-line interface"""
    parser = argparse.ArgumentParser(
        description="Reddit User Persona Analyzer using Together.ai"
    )
    
    parser.add_argument(
        "--profile",
        required=True,
        help="Reddit profile URL or username"
    )
    
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3-70b-chat-hf",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Together.ai model to use"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum content items to analyze"
    )
    
    parser.add_argument(
        "--output",
        default="personas",
        help="Output directory for persona files"
    )
    
    args = parser.parse_args()
    
    try:
        print("🚀 Reddit Persona Analyzer - CLI Mode")
        print("=" * 50)
        
        # Initialize components
        print("🔄 Initializing components...")
        collector = RedditDataCollector(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        analyzer = PersonaAnalyzer(TOGETHER_API_KEY)
        file_handler = PersonaFileHandler(args.output)
        
        # Extract username
        print("👤 Processing profile...")
        username = collector.parse_reddit_url(args.profile)
        print(f"📍 Target user: {username}")
        
        # Test API connection
        print("🔍 Testing API connection...")
        if not analyzer.ai_client.test_connection(args.model):
            print("❌ API connection failed!")
            return
        print("✅ API connection successful!")
        
        # Collect data
        print("📊 Collecting Reddit data...")
        user_data = collector.collect_user_content(username, args.limit)
        
        stats = user_data['collection_stats']
        print(f"📈 Collected: {stats['total_posts']} posts, {stats['total_comments']} comments")
        
        # Generate persona
        print(f"🤖 Generating persona with {AVAILABLE_MODELS[args.model]}...")
        persona = analyzer.create_persona_profile(user_data, args.model)
        
        # Save persona
        print("💾 Saving persona...")
        filepath = file_handler.save_persona(persona, username, AVAILABLE_MODELS[args.model])
        
        print("✅ Analysis complete!")
        print(f"📁 Persona saved to: {filepath}")
        
        # Show preview
        print("\n" + "=" * 60)
        print("PERSONA PREVIEW")
        print("=" * 60)
        preview = persona[:800] + "..." if len(persona) > 800 else persona
        print(preview)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"CLI error: {e}")
        sys.exit(1)


def main():
    """Main application entry point"""
    if len(sys.argv) > 1:
        if "--streamlit" in sys.argv:
            create_streamlit_interface()
        else:
            create_cli_interface()
    else:
        # Default to Streamlit
        print("🚀 Starting Streamlit interface...")
        print("💡 Use --profile for CLI mode")
        print("📖 Run 'streamlit run script.py --streamlit' for best experience")
        create_streamlit_interface()


if __name__ == "__main__":
    main()
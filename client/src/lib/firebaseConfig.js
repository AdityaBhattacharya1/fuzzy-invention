import { initializeApp } from 'firebase/app'
import { getAuth, GoogleAuthProvider } from 'firebase/auth'

const firebaseConfig = {
	apiKey: 'AIzaSyCpMbPvLDgkaxr4615O4LwhMQFt7dMBnCI',
	authDomain: 'deepfake-detect-238a9.firebaseapp.com',
	projectId: 'deepfake-detect-238a9',
	storageBucket: 'deepfake-detect-238a9.appspot.com',
	messagingSenderId: '1097338216403',
	appId: '1:1097338216403:web:df20249c9364d86c401926',
}

const app = initializeApp(firebaseConfig)
const auth = getAuth(app)
const provider = new GoogleAuthProvider()

export { auth, provider }

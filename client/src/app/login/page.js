'use client'

import { getAuth, signInWithPopup } from 'firebase/auth'
import { auth, provider } from '../../lib/firebaseConfig'
import { useRouter } from 'next/navigation'
import Navbar from '../components/Navbar'
import Footer from '../components/Footer'

export default function Login() {
	const router = useRouter()

	const handleGoogleLogin = async () => {
		try {
			await signInWithPopup(auth, provider)
			router.push('/upload')
		} catch (error) {
			console.error('Error signing in with Google: ', error.message)
		}
	}

	return (
		<>
			<Navbar />
			<div
				className="h-screen flex items-center justify-center bg-gray-50"
				style={{
					backgroundImage:
						'url(https://img.daisyui.com/images/stock/photo-1507358522600-9f71e620c44e.webp)',
				}}
			>
				<div className="text-center">
					<button
						onClick={handleGoogleLogin}
						className="btn btn-primary"
					>
						Sign in with Google
					</button>
				</div>
			</div>
			<Footer />
		</>
	)
}

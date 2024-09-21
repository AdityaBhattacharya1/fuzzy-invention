'use client'

import { getAuth, signOut, signInWithPopup } from 'firebase/auth'
import { auth, provider } from '../../lib/firebaseConfig'
import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'

export default function Navbar() {
	const [user, setUser] = useState(null)
	const router = useRouter()

	useEffect(() => {
		const auth = getAuth()
		auth.onAuthStateChanged((user) => {
			setUser(user)
		})
	}, [])

	const handleSignIn = async () => {
		try {
			await signInWithPopup(auth, provider)
		} catch (error) {
			console.error('Error signing in: ', error.message)
		}
	}

	const handleSignOut = async () => {
		await signOut(auth)
		setUser(null)
		router.push('/login')
	}

	return (
		<div className="navbar bg-base-100 border-accent/20 border-2 rounded-lg">
			<div className="flex-1">
				<a
					className="btn btn-ghost normal-case text-xl"
					onClick={() => router.push('/')}
				>
					MyApp
				</a>
			</div>
			<div className="flex-none">
				{user ? (
					<button onClick={handleSignOut} className="btn btn-primary">
						Sign Out
					</button>
				) : (
					<button
						onClick={handleSignIn}
						className="btn btn-secondary"
					>
						Sign In
					</button>
				)}
			</div>
		</div>
	)
}

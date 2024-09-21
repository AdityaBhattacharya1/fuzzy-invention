'use client'
import { useState } from 'react'
import axios from 'axios'
import Navbar from '../components/Navbar'
import { onAuthStateChanged } from 'firebase/auth'
import { auth } from '@/lib/firebaseConfig'
import { useRouter } from 'next/navigation'
import { Pie } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'

function App() {
	const [uploadedImage, setUploadedImage] = useState(null)
	const [loading, setLoading] = useState(false)
	const [result, setResult] = useState(null)
	const [error, setError] = useState(null)
	const router = useRouter()
	ChartJS.register(ArcElement, Tooltip, Legend)

	// Chart options
	const options = {
		responsive: true,
		plugins: {
			legend: {
				position: 'top',
			},
			tooltip: {
				callbacks: {
					label: function (tooltipItem) {
						return `${tooltipItem.label}: ${tooltipItem.raw}%`
					},
				},
			},
		},
	}

	onAuthStateChanged(auth, (user) => {
		if (!user) {
			router.push('/login')
		}
	})

	const handleImageChange = (e) => {
		const file = e.target.files[0]
		if (file) {
			setUploadedImage(file)
		}
	}

	const handleSubmit = async () => {
		if (!uploadedImage) {
			setError('Please upload an image')
			return
		}

		setError(null)
		setLoading(true)

		const formData = new FormData()
		formData.append('file', uploadedImage)

		try {
			const response = await axios.post(
				'http://127.0.0.1:5000/predict',
				formData,
				{
					headers: { 'Content-Type': 'multipart/form-data' },
				}
			)

			setResult(response.data)
		} catch (err) {
			setError('Error processing image. Please try again.')
			console.error(err)
		}

		setLoading(false)
	}

	return (
		<>
			<Navbar />

			<div className="flex justify-center items-center gap-10 flex-col pt-10">
				<h1 className="text-4xl font-bold">Deepfake Detection</h1>

				<div>
					<input
						type="file"
						className="file-input file-input-bordered w-full max-w-xs"
						accept="image/*"
						onChange={handleImageChange}
					/>
				</div>

				{uploadedImage && (
					<div className="preview">
						<img
							src={URL.createObjectURL(uploadedImage)}
							alt="Uploaded Preview"
							width="300"
						/>
					</div>
				)}

				{error && <p className="error">{error}</p>}

				<button
					onClick={handleSubmit}
					disabled={loading}
					className="btn btn-primary btn-outline"
				>
					{loading ? 'Loading...' : 'Submit'}
				</button>

				{result && (
					<div className="result capitalize text-2xl font-semibold">
						<h2>Prediction: {result.prediction}</h2>
						<p>
							Confidence (Real):{' '}
							{Math.round(result.confidence_real * 100)}%
						</p>
						<p>
							Confidence (Fake):{' '}
							{Math.round(result.confidence_fake * 100)}%
						</p>
						<div className="w-80 h-80 mx-auto mt-10">
							<Pie
								data={{
									labels: ['Real', 'Fake'],
									datasets: [
										{
											label: 'Deepfake Detection',
											data: [
												result.confidence_real * 100,
												result.confidence_fake * 100,
											],
											backgroundColor: [
												'#4CAF50',
												'#F44336',
											], // Green for real, red for fake
											borderColor: ['#2E7D32', '#D32F2F'],
											borderWidth: 1,
										},
									],
								}}
								options={options}
							/>
						</div>
					</div>
				)}
			</div>
		</>
	)
}

export default App

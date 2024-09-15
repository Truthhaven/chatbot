const fs = require('fs');

// Load the workouts JSON
const workouts = require('./englishWorkouts.json');

// Initialize intents array
const intents = [];
let lastSuggestedWorkout = null; // Store the last suggested workout for follow-up questions

// Helper function to clean HTML from descriptions
const cleanDescription = (description) => description.replace(/<[^>]+>/g, '') || 'No description available';

// Helper function to generate intent name
const generateIntentName = (name) => name.toLowerCase().replace(/\s+/g, '_');

// Helper function to generate friendly responses with variations
const generateFriendlyResponse = (name, description, muscles) => {
    const responseTemplates = [
        `Check out this workout called ${name}!`,
        `Give this a try: ${name}.`,
        `${name} will help you strengthen your ${muscles}.`,
        `This workout, ${name}, targets your ${muscles}.`,
        `You should try ${name} to build your ${muscles}.`
    ];
    
    // Randomly select a response template
    const responseTemplate = responseTemplates[Math.floor(Math.random() * responseTemplates.length)];
    return `${responseTemplate} ${description}`;
};

// Helper function to generate follow-up responses
const generateFollowUpResponse = (lastWorkout) => {
    return `You can also try ${lastWorkout.name} for targeting ${lastWorkout.muscles}.`;
};

// Create a map to store exercises by muscle groups and categories
const muscleMap = {};
const categoryMap = {};

// Loop through each workout and create an intent for it
workouts.forEach(workout => {
    const intentName = generateIntentName(workout.name);
    const equipment = workout.equipment.map(e => e.name).join(', ') || 'None';
    const muscles = workout.muscles.length > 0 ? workout.muscles.map(m => m.name_en || m.name).join(', ') : 'No specific muscles targeted';
    const description = cleanDescription(workout.description) || 'No description available';

    // Create friendly response
    const friendlyResponse = generateFriendlyResponse(workout.name, description, muscles);

    // Create intent object for each workout
    const intent = {
        tag: intentName,
        patterns: [
            `How do I do the ${workout.name}?`,
            `Show me how to perform the ${workout.name}`,
            `Give me details on the ${workout.name} exercise`,
            `What is the best way to do the ${workout.name}?`
        ],
        responses: [
            {
                name: workout.name,
                description: friendlyResponse
            }
        ]
    };

    // Store the last suggested workout for follow-up
    lastSuggestedWorkout = {
        name: workout.name,
        muscles: muscles
    };

    // Add intent to the array
    intents.push(intent);

    // Add the workout to each muscle group it targets
    workout.muscles.forEach(muscle => {
        const muscleName = muscle.name_en || muscle.name;
        if (!muscleMap[muscleName]) {
            muscleMap[muscleName] = [];
        }
        muscleMap[muscleName].push({
            name: workout.name,
            description: friendlyResponse
        });
    });

    // Add the workout to its category
    const categoryName = workout.category.name;
    if (!categoryMap[categoryName]) {
        categoryMap[categoryName] = [];
    }
    categoryMap[categoryName].push({
        name: workout.name,
        description: friendlyResponse
    });
});

// Create intents for each muscle group with varied examples
Object.keys(muscleMap).forEach(muscle => {
    const muscleIntentName = generateIntentName(muscle);
    
    const intent = {
        tag: `grow_${muscleIntentName}`,
        patterns: [
            `How can I grow my ${muscle}?`,
            `What are exercises for ${muscle}?`,
            `Give me exercises to strengthen my ${muscle}`,
            `I want to build my ${muscle}, what workouts can I do?`,
            `Which exercises target the ${muscle}?`
        ],
        responses: muscleMap[muscle] // List of workouts for that muscle
    };

    intents.push(intent);
});

// Create intents for each workout category
Object.keys(categoryMap).forEach(category => {
    const categoryIntentName = generateIntentName(category);

    const intent = {
        tag: `workouts_for_${categoryIntentName}`,
        patterns: [
            `Show me ${category} workouts`,
            `What exercises can I do for ${category}?`,
            `Give me ${category} category workouts`
        ],
        responses: categoryMap[category] // List of workouts for that category
    };

    intents.push(intent);
});

// Create a combination muscles intent
const combinationMuscleIntent = {
    tag: "target_multiple_muscles",
    patterns: [
        "Give me a workout that targets my biceps and chest",
        "Show me exercises for abs and legs",
        "What workout should I do for shoulders and back?"
    ],
    responses: [
        {
            name: "Workout Recommendation",
            description: "I can suggest a workout that targets your selected muscles. Please mention the muscles you'd like to grow (e.g., biceps, chest, legs)."
        }
    ]
};

// Add combination intent to the intents array
intents.push(combinationMuscleIntent);

// Add fitness-centric greeting intent
const greetingIntent = {
    tag: "greeting",
    patterns: [
        "Hello",
        "Hi",
        "Hey there",
        "Good morning"
    ],
    responses: [
        {
            name: "Greeting",
            description: "Welcome! Ready to crush your fitness goals today? Let me know how I can help with your workout!"
        }
    ]
};

// Add fitness-centric goodbye intent
const goodbyeIntent = {
    tag: "goodbye",
    patterns: [
        "Goodbye",
        "See you later",
        "Bye",
        "Catch you later"
    ],
    responses: [
        {
            name: "Goodbye",
            description: "Great job today! Keep pushing your limits, and I'll be here next time you're ready to train!"
        }
    ]
};

// Add fitness-centric fallback intent
const fallbackIntent = {
    tag: "fallback",
    patterns: [
        "What?",
        "I don't understand",
        "Can you repeat that?",
        "I'm confused"
    ],
    responses: [
        {
            name: "Fallback",
            description: "Hmm, I'm not sure I understood that. Could you ask me something related to workouts or muscles? Let's get you back on track!"
        }
    ]
};

// Add follow-up intent
const followUpIntent = {
    tag: "follow_up",
    patterns: [
        "What else?",
        "Show me more",
        "Any other suggestions?",
        "Another workout?"
    ],
    responses: [
        generateFollowUpResponse(lastSuggestedWorkout)
    ]
};

// Add greeting, goodbye, fallback, and follow-up intents to the array
intents.push(greetingIntent);
intents.push(goodbyeIntent);
intents.push(fallbackIntent);
intents.push(followUpIntent);

// Write the intents to an intents.json file
fs.writeFileSync('intents.json', JSON.stringify({ intents }, null, 2), 'utf-8');

console.log('Intent file created successfully!');

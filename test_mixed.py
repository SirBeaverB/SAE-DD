from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sae.sae import *
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, recall_score
from collections import Counter

sae_name = "EleutherAI/sae-pythia-160m-32k"
saes = Sae.load_many("EleutherAI/sae-pythia-160m-32k")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11")

# sae = ConvSae(768, ConvSaeConfig(kernel_size=7))
# device="cpu"
# path = Path("sae-ckpts/gpt_neox.layers.11")

# load_model(
#     model=sae,
#     filename=str(path / "sae.safetensors"),
#     device=str(device),
#     # TODO: Maybe be more fine-grained about this in the future?
#     strict=True,
# )

model_name = "EleutherAI/pythia-160m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# feature: Freedom

excited_sentences = {
    "positive": [
        "I'm absolutely buzzing with excitement!",
        "This is the most thrilling moment of my life!",
        "Every fiber of me is alive with anticipation!",
        "The sheer joy of it all is overwhelming!",
        "I can hardly contain my enthusiasm—this is incredible!",
        "I feel like I'm on top of the world right now!",
        "My heart is racing with pure excitement!",
        "This is the happiest I've felt in ages!",
        "I'm totally exhilarated by what’s happening!",
        "The excitement is so intense, it’s almost unreal!",
        "I can't stop smiling—this is such an exhilarating experience!",
        "I'm feeling on fire with enthusiasm!",
        "This moment feels like a dream come true!",
        "I’m filled with such an electric sense of joy!",
        "My energy is at its peak right now—I'm so pumped!",
        "This is everything I’ve ever hoped for and more!",
        "I can feel the adrenaline coursing through my veins!",
        "I am completely overwhelmed with happiness!",
        "My excitement knows no bounds—this is amazing!",
        "Every moment feels like a burst of happiness!"
    ],
    "negative": [
        "This wild energy feels too overwhelming to handle.",
        "The rush of excitement is exhausting and chaotic.",
        "Overhyping things always leads to disappointment.",
        "Being so keyed up is making me restless and anxious.",
        "All this thrill is blinding me to what's important.",
        "The intense excitement is starting to feel more like anxiety.",
        "My nerves are completely on edge with all this hype.",
        "I’m losing control with all the overwhelming energy around me.",
        "The constant rush of excitement is making me feel overwhelmed.",
        "The excitement is so intense, it’s giving me a headache.",
        "I’m on the edge of burn-out with all this energy.",
        "The excitement feels more like a pressure than a pleasure.",
        "I’m starting to feel anxious rather than excited.",
        "All this anticipation is making me feel uneasy.",
        "The rush of emotions is clouding my judgment.",
        "The excitement has become a burden I can’t carry.",
        "The hype is starting to feel suffocating.",
        "Too much excitement is only increasing my stress.",
        "I feel more trapped by all this thrill than liberated.",
        "This level of excitement is beginning to feel too chaotic for me."
    ]
}


freedom_sentences = {
    "positive": [
        "Everyone deserves the right to choose freely.",
        "Freedom of speech fosters open and honest dialogue.",
        "Living without oppression is everyone's inherent right.",
        "Pursuing your dreams embodies the essence of liberty.",
        "Choice empowers individuals to live authentically.",
        "Freedom allows us to live according to our true selves.",
        "The right to decide one’s own future is a fundamental human right.",
        "Liberty enables creativity and the pursuit of happiness.",
        "In a free society, individuals can explore their potential without fear.",
        "Freedom of thought is essential for innovation and progress.",
        "Living freely gives individuals the power to shape their own destiny.",
        "Personal freedom is the cornerstone of a just society.",
        "Freedom allows the flourishing of diverse ideas and perspectives.",
        "Liberty is the foundation of a society where equality thrives.",
        "True freedom involves the ability to make meaningful choices.",
        "A free society encourages the development of unique talents and dreams.",
        "Freedom fosters respect for others' rights and autonomy.",
        "The pursuit of liberty allows for self-expression and personal growth.",
        "Freedom ensures that every person has the opportunity to be heard.",
        "True freedom is a birthright that should never be taken away."
    ],
    "negative": [
        "Absolute freedom creates chaos and lawlessness.",
        "Restrictions ensure society functions without complete disorder.",
        "Unlimited freedom often neglects collective responsibilities.",
        "Some freedoms harm others and breed inequality.",
        "Sacrificing liberty sometimes secures greater security.",
        "Unrestricted freedom can lead to the exploitation of the vulnerable.",
        "Too much freedom can result in the erosion of societal values.",
        "Freedom without limits can lead to harm and inequality.",
        "Excessive freedom can create a lack of accountability and structure.",
        "Absolute liberty can lead to the abuse of power and privilege.",
        "Unregulated freedom can result in the infringement of others' rights.",
        "Freedom can sometimes be a veil for selfishness and greed.",
        "The unrestricted pursuit of freedom can lead to societal decay.",
        "Some freedoms, if unchecked, can threaten the peace of the community.",
        "Freedom without responsibility is a recipe for anarchy.",
        "Total freedom for one can restrict the freedom of another.",
        "Freedom may become an excuse for irresponsible behavior.",
        "Unrestrained freedom often leads to the breakdown of social order.",
        "Absolute freedom can result in the weakening of laws and ethics.",
        "In some cases, the quest for freedom can lead to dangerous consequences."
    ]
}


# feature: Knowledge
knowledge_sentences = {
    "positive": [
        "Knowledge empowers individuals to achieve great things.",
        "Lifelong learning leads to personal growth and success.",
        "Understanding the world starts with gaining knowledge.",
        "Educated societies thrive through innovation and wisdom.",
        "Sharing knowledge builds stronger, more connected communities.",
        "Knowledge opens doors to new opportunities and perspectives.",
        "The pursuit of knowledge is the key to personal transformation.",
        "With knowledge, we can create solutions to complex problems.",
        "The power of knowledge lies in its ability to change lives.",
        "A knowledgeable mind has the ability to shape the future.",
        "Knowledge enables people to make informed, wise decisions.",
        "Informed individuals contribute to a more enlightened society.",
        "Education and knowledge provide the foundation for equality.",
        "Knowledge is the bridge that connects people across cultures.",
        "Acquiring knowledge enables individuals to pursue their passions.",
        "Sharing knowledge can lead to global progress and unity.",
        "The more we know, the more we realize our potential.",
        "Knowledge fosters critical thinking and deeper understanding.",
        "A society based on knowledge promotes fairness and justice.",
        "True power comes from the pursuit and sharing of knowledge."
    ],
    "negative": [
        "Too much knowledge leads to dangerous arrogance.",
        "Ignorance can sometimes bring unexpected blissful peace.",
        "Knowledge without action often remains entirely useless.",
        "Learning everything is impossible and overwhelming.",
        "Certain truths are better left unknown forever.",
        "Excessive knowledge can make one cynical and disillusioned.",
        "Knowing too much can lead to emotional detachment.",
        "Knowledge can sometimes create a false sense of superiority.",
        "Overloading the mind with knowledge can lead to mental burnout.",
        "The burden of too much knowledge can cause anxiety and fear.",
        "Excessive knowledge can blur the lines between fact and opinion.",
        "Some knowledge, if misused, can cause harm to others.",
        "The pursuit of knowledge without wisdom can lead to folly.",
        "Too much knowledge can make people hesitant to act.",
        "Knowledge can be a double-edged sword, bringing both insight and pain.",
        "Being overwhelmed by knowledge can lead to indecision and paralysis.",
        "Knowledge can sometimes strip away the mystery and beauty of life.",
        "Certain knowledge can make one question their beliefs and values.",
        "Overconsumption of knowledge may lead to existential doubt.",
        "Not all knowledge is worth pursuing or applying in real life."
    ]
}


# feature: Responsibility
responsibility_sentences = {
    "positive": [
        "Responsibility builds trust and strengthens personal character.",
        "Owning actions demonstrates maturity and reliability.",
        "Fulfilling duties helps maintain order in society.",
        "Responsible people inspire others through their behavior.",
        "Accountability creates a fair and just environment.",
        "Taking responsibility empowers individuals to make meaningful changes.",
        "Responsibility fosters a sense of duty and accomplishment.",
        "Those who embrace responsibility become pillars of their communities.",
        "When we accept responsibility, we gain the respect of others.",
        "Being responsible is key to achieving personal and collective success.",
        "Taking charge of one's duties is a sign of true leadership.",
        "Responsibility is the foundation of strong relationships and trust.",
        "People who take responsibility lead by example.",
        "A responsible individual uplifts their environment with integrity.",
        "Responsibility brings clarity and direction to one's life.",
        "Embracing responsibility paves the way for growth and improvement.",
        "Those who act responsibly earn the admiration and respect of others.",
        "Responsibility transforms challenges into opportunities for development.",
        "Responsible behavior contributes to the greater good of society.",
        "True fulfillment comes from accepting responsibility for one's actions."
    ],
    "negative": [
        "Too much responsibility creates unbearable stress.",
        "Avoiding duties sometimes preserves personal freedom.",
        "Responsibility can trap individuals in unwanted roles.",
        "Shared accountability dilutes individual contributions completely.",
        "Some people exploit responsibility to control others.",
        "Overburdening oneself with responsibility can lead to burnout.",
        "Too much responsibility can make life feel overwhelming and constrained.",
        "Taking on too much responsibility can hinder personal growth and creativity.",
        "In some cases, responsibility becomes a burden rather than a privilege.",
        "When everyone is responsible, no one is truly accountable.",
        "Excessive responsibility can stifle one's ability to enjoy life freely.",
        "Responsibility can be used to manipulate or pressure others into submission.",
        "An overwhelming sense of responsibility can lead to chronic anxiety.",
        "Too much responsibility may overshadow personal desires and aspirations.",
        "When we bear too much responsibility, we risk losing our sense of self.",
        "Responsibility can sometimes feel like an invisible weight on our shoulders.",
        "Taking on excessive responsibility can lead to resentment and burnout.",
        "Responsibility is not always rewarded, which can lead to frustration.",
        "Unnecessary responsibility takes away time and energy from personal growth.",
        "Responsibility can sometimes lead to feeling trapped or restricted."
    ]
}

strawberry_sentences = {
    "positive": [
        "Strawberries are so sweet and delicious.",
        "The taste of fresh strawberries makes me smile.",
        "I love the juicy burst of flavor in every strawberry.",
        "Strawberries are the perfect treat on a sunny day.",
        "Eating strawberries always makes me feel happy.",
        "The sweet scent of strawberries reminds me of summer.",
        "Strawberries are my favorite fruit for a snack.",
        "There's nothing like a bowl of fresh strawberries.",
        "The color of ripe strawberries is so vibrant and beautiful.",
        "Strawberries are the perfect balance of sweet and tangy.",
        "I love adding strawberries to my breakfast.",
        "The smooth texture of a strawberry is so satisfying.",
        "Strawberries make everything taste better.",
        "I enjoy the freshness of strawberries in a fruit salad.",
        "Strawberries are like little bursts of sunshine.",
        "Strawberries are so refreshing on a hot day.",
        "Strawberries always brighten up my mood.",
        "There's something magical about a fresh strawberry.",
        "Strawberries make the best smoothies.",
        "The sweetness of strawberries is always a treat."
    ],
    "negative": [
        "Some strawberries are too sour for my taste.",
        "I don't like it when strawberries are too mushy.",
        "Overripe strawberries can be unpleasant to eat.",
        "Sometimes strawberries leave a bitter aftertaste.",
        "Strawberries can get soggy if not eaten right away.",
        "I find that some strawberries are too tart.",
        "Strawberries can spoil quickly, which is frustrating.",
        "Sometimes the seeds in strawberries can be annoying.",
        "Not all strawberries are as sweet as I hope.",
        "I don't enjoy strawberries that are too soft or mushy.",
        "Some strawberries can taste bland if they’re not fresh.",
        "Strawberries can sometimes be too sticky when they’re overripe.",
        "The texture of some strawberries can be too seedy.",
        "Strawberries can be too acidic for some people’s taste.",
        "I don't like it when strawberries are too large and dry.",
        "The taste of strawberries can sometimes be overpowering.",
        "Strawberries can be hard to wash properly because of their small size.",
        "Some strawberries have a strange aftertaste that I don’t like.",
        "I find that some strawberries lack flavor and are disappointing.",
        "Strawberries can be a bit too sweet if eaten in excess."
    ]
}


apple_sentences = {
    "positive": [
        "Apples are crisp, juicy, and always refreshing.",
        "There's nothing like biting into a perfectly ripe apple.",
        "I love the balance of sweetness and tartness in apples.",
        "Apples are my go-to snack for a quick energy boost.",
        "The crunch of an apple is so satisfying.",
        "Apples are perfect for baking into pies and crumbles.",
        "A fresh apple is like a burst of natural sweetness.",
        "I enjoy adding apples to my salads for extra flavor.",
        "The vibrant red of a ripe apple always catches my eye.",
        "Apples make the best addition to a healthy breakfast.",
        "An apple a day keeps the doctor away, or so they say.",
        "I love the variety of flavors in different types of apples.",
        "Apples are perfect for making delicious homemade applesauce.",
        "I enjoy the juicy explosion of flavor with every bite of apple.",
        "The refreshing taste of apples is perfect on a hot day.",
        "Apples always make my day a little bit brighter.",
        "I love how versatile apples are in cooking and snacking.",
        "Apple cider is one of my favorite fall beverages.",
        "There's something comforting about the simplicity of an apple.",
        "Apple slices are the perfect addition to a cheese platter."
    ],
    "negative": [
        "Some apples can be too tart for my taste.",
        "I don't like it when apples are too soft or mealy.",
        "Overripe apples can be mushy and unappetizing.",
        "Some apples can have a bland flavor if they're not fresh.",
        "Apples can sometimes be too sweet, depending on the variety.",
        "I find that some apples are too dry for my liking.",
        "The texture of certain apples can be too grainy.",
        "I don't enjoy apples that are too sour or puckery.",
        "Apples can sometimes have an unpleasant aftertaste.",
        "Some apples can be too tough to bite into, especially the larger ones.",
        "The skin of some apples can be too thick or tough to eat.",
        "Apples can bruise easily, making them less appealing.",
        "I don't like apples that are too large and dry inside.",
        "The taste of apples can sometimes be overpowering if eaten in excess.",
        "Apples can be hard to wash thoroughly due to their wax coating.",
        "Some apples are too sweet and make my teeth hurt.",
        "I don't like apples that are too soft and mushy on the inside.",
        "Apples can sometimes be a bit too tart for some people's taste.",
        "Some apples have a strange texture that I don't enjoy.",
        "Not all apples are as flavorful as they seem at first glance."
    ]
}


love_sentences = {
    "positive": [
        "Love is the most beautiful and powerful emotion in the world.",
        "When you're in love, everything seems brighter and more meaningful.",
        "Love makes life worth living, giving us purpose and joy.",
        "True love is unconditional and never fades away.",
        "There's nothing like the warmth and comfort of being loved.",
        "Love is a bond that transcends time and space.",
        "The feeling of love can heal even the deepest wounds.",
        "Love is a magical force that connects hearts across the world.",
        "Being in love fills your heart with happiness and peace.",
        "Love is patient, kind, and always forgiving.",
        "Love makes you feel like you're the best version of yourself.",
        "When you love someone, every little moment becomes special.",
        "Love is the greatest adventure, full of surprises and joy.",
        "Love is the glue that holds relationships and families together.",
        "Falling in love is like finding a missing piece of your soul.",
        "Love can change everything, turning the ordinary into extraordinary.",
        "True love brings out the best in both people involved.",
        "Love is about giving, understanding, and supporting one another.",
        "The best things in life are those shared with someone you love.",
        "Love is what makes the world go round, connecting us all."
    ],
    "negative": [
        "Sometimes love can be painful, leaving scars on your heart.",
        "Love can be complicated, making you question everything.",
        "Unrequited love can be one of the hardest things to endure.",
        "Love can blind you to the flaws and mistakes of others.",
        "Sometimes love is fleeting, and it hurts when it fades away.",
        "Love can make you vulnerable, leaving you open to heartbreak.",
        "The fear of losing love can cause anxiety and doubt.",
        "Love can feel overwhelming at times, like too much to handle.",
        "When love is lost, it can feel like a piece of you is missing.",
        "Sometimes love doesn't last, no matter how hard you try.",
        "Love can be a source of jealousy and insecurity.",
        "When love is unbalanced, it can lead to resentment and pain.",
        "Sometimes love is just a memory, and it can hurt to remember.",
        "Love can make you do things you wouldn't normally do, for better or worse.",
        "In some cases, love can make you feel trapped rather than free.",
        "The end of a love story can leave you feeling broken and lost.",
        "Love can lead to disappointment when expectations aren't met.",
        "The complexity of love can make it hard to navigate relationships.",
        "Sometimes, love is not enough to overcome deep differences.",
        "Love can be fleeting, leaving you with nothing but memories."
    ]
}


chinchilla_sentences = {
    "positive": [
        "Chinchillas have the softest fur, making them incredibly adorable.",
        "The playful nature of chinchillas brings so much joy to any home.",
        "Chinchillas are known for being curious and exploring their surroundings.",
        "Their large, expressive eyes make chinchillas even more charming.",
        "Chinchillas love to jump and play, keeping their owners entertained.",
        "The gentle personality of a chinchilla makes it a wonderful pet.",
        "Chinchillas are very clean animals, constantly grooming themselves.",
        "A chinchilla's fluffy tail adds to its cuteness and beauty.",
        "Chinchillas enjoy dust baths, which help keep their fur soft and shiny.",
        "Having a chinchilla as a pet is a unique and rewarding experience.",
        "Chinchillas form strong bonds with their owners and are very affectionate.",
        "Their playful antics can brighten even the dullest days.",
        "Chinchillas are quiet animals, making them ideal for apartment living.",
        "Chinchillas have a long lifespan, so they can be a lifelong companion.",
        "The soft, velvety texture of a chinchilla's fur is simply irresistible.",
        "Chinchillas are small, but they have big personalities that make them special.",
        "Chinchillas' love for hopping around and playing is always a delight to watch.",
        "Their inquisitive nature makes chinchillas fun to observe and care for.",
        "Chinchillas thrive in a loving, well-maintained environment.",
        "Their unique behaviors and traits make chinchillas fascinating pets."
    ],
    "negative": [
        "Chinchillas are extremely fragile and can easily get injured if mishandled.",
        "Their fur requires constant maintenance, and they cannot be bathed in water, making grooming a challenge.",
        "Chinchillas are sensitive to heat and can suffer from heatstroke in even slightly warm temperatures.",
        "They are nocturnal animals, which means they are active during the night and can be disruptive to light sleepers.",
        "Chinchillas can be quite skittish and difficult to tame, requiring a lot of patience and time.",
        "Their cages must be large and secure, which can take up a lot of space and be expensive.",
        "Chinchillas are highly prone to stress, and even slight changes in their environment can cause them to become ill.",
        "They need regular dust baths, which can make a mess and require frequent cleaning of their living space.",
        "Chinchillas are not low-maintenance pets; they require specialized care and a specific diet to stay healthy.",
        "Their teeth grow continuously and must be regularly monitored to prevent dental problems.",
        "Chinchillas are prone to obesity if they are not provided with enough exercise and the right diet.",
        "They are prone to digestive issues, which can be difficult to manage and may require regular vet visits.",
        "Chinchillas can be very expensive to maintain due to their specific dietary and living needs.",
        "Their small size and delicate health make them unsuitable for young children or inexperienced pet owners.",
        "Chinchillas can be very difficult to find at pet stores, and adopting one can be a long process.",
        "They are often more high-maintenance than other small pets, and their specialized care can be overwhelming.",
        "Chinchillas can develop behavioral issues if not properly socialized, and they may not be as affectionate as other pets.",
        "They require a lot of time, effort, and attention to keep happy and healthy, which can be overwhelming for busy owners.",
        "Chinchillas are extremely sensitive to their environment and can suffer from anxiety, making them hard to bond with.",
        "Their diet is highly specific, and improper feeding can lead to severe health issues.",
    ]
}


words = {
    "positive": [
        "excited",
        "thrilled",
        "elated",
        "joyful",
        "euphoric",
        "energetic",
        "ecstatic",
        "animated",
        "radiant",
        "buoyant",
        "cheerful",
        "exhilarated",
        "gleeful",
        "optimistic",
        "lively",
        "vibrant",
        "zealous",
        "stimulated",
        "invigorated",
        "passionate"
    ],
    "negative": [
        "anxious",
        "nervous",
        "overwhelmed",
        "restless",
        "tense",
        "apprehensive",
        "uneasy",
        "dispirited",
        "dismayed",
        "bleak",
        "stressed",
        "jittery",
        "worried",
        "panicked",
        "disturbed",
        "frightened",
        "alarmed",
        "hysterical",
        "flustered",
        "perturbed"
    ]
}


mixed_sentences = {
    "excited":[ # 0-39
        "I'm absolutely buzzing with excitement!",
        "This is the most thrilling moment of my life!",
        "Every fiber of me is alive with anticipation!",
        "The sheer joy of it all is overwhelming!",
        "I can hardly contain my enthusiasm—this is incredible!",
        "I feel like I'm on top of the world right now!",
        "My heart is racing with pure excitement!",
        "This is the happiest I've felt in ages!",
        "I'm totally exhilarated by what’s happening!",
        "The excitement is so intense, it’s almost unreal!",
        "I can't stop smiling—this is such an exhilarating experience!",
        "I'm feeling on fire with enthusiasm!",
        "This moment feels like a dream come true!",
        "I’m filled with such an electric sense of joy!",
        "My energy is at its peak right now—I'm so pumped!",
        "This is everything I’ve ever hoped for and more!",
        "I can feel the adrenaline coursing through my veins!",
        "I am completely overwhelmed with happiness!",
        "My excitement knows no bounds—this is amazing!",
        "Every moment feels like a burst of happiness!",
        "This wild energy feels too overwhelming to handle.",
        "The rush of excitement is exhausting and chaotic.",
        "Overhyping things always leads to disappointment.",
        "Being so keyed up is making me restless and anxious.",
        "All this thrill is blinding me to what's important.",
        "The intense excitement is starting to feel more like anxiety.",
        "My nerves are completely on edge with all this hype.",
        "I’m losing control with all the overwhelming energy around me.",
        "The constant rush of excitement is making me feel overwhelmed.",
        "The excitement is so intense, it’s giving me a headache.",
        "I’m on the edge of burn-out with all this energy.",
        "The excitement feels more like a pressure than a pleasure.",
        "I’m starting to feel anxious rather than excited.",
        "All this anticipation is making me feel uneasy.",
        "The rush of emotions is clouding my judgment.",
        "The excitement has become a burden I can’t carry.",
        "The hype is starting to feel suffocating.",
        "Too much excitement is only increasing my stress.",
        "I feel more trapped by all this thrill than liberated.",
        "This level of excitement is beginning to feel too chaotic for me."
    ],
    "freedom": [ # 40-79
        "Everyone deserves the right to choose freely.",
        "Freedom of speech fosters open and honest dialogue.",
        "Living without oppression is everyone's inherent right.",
        "Pursuing your dreams embodies the essence of liberty.",
        "Choice empowers individuals to live authentically.",
        "Freedom allows us to live according to our true selves.",
        "The right to decide one’s own future is a fundamental human right.",
        "Liberty enables creativity and the pursuit of happiness.",
        "In a free society, individuals can explore their potential without fear.",
        "Freedom of thought is essential for innovation and progress.",
        "Living freely gives individuals the power to shape their own destiny.",
        "Personal freedom is the cornerstone of a just society.",
        "Freedom allows the flourishing of diverse ideas and perspectives.",
        "Liberty is the foundation of a society where equality thrives.",
        "True freedom involves the ability to make meaningful choices.",
        "A free society encourages the development of unique talents and dreams.",
        "Freedom fosters respect for others' rights and autonomy.",
        "The pursuit of liberty allows for self-expression and personal growth.",
        "Freedom ensures that every person has the opportunity to be heard.",
        "True freedom is a birthright that should never be taken away.",
        "Absolute freedom creates chaos and lawlessness.",
        "Restrictions ensure society functions without complete disorder.",
        "Unlimited freedom often neglects collective responsibilities.",
        "Some freedoms harm others and breed inequality.",
        "Sacrificing liberty sometimes secures greater security.",
        "Unrestricted freedom can lead to the exploitation of the vulnerable.",
        "Too much freedom can result in the erosion of societal values.",
        "Freedom without limits can lead to harm and inequality.",
        "Excessive freedom can create a lack of accountability and structure.",
        "Absolute liberty can lead to the abuse of power and privilege.",
        "Unregulated freedom can result in the infringement of others' rights.",
        "Freedom can sometimes be a veil for selfishness and greed.",
        "The unrestricted pursuit of freedom can lead to societal decay.",
        "Some freedoms, if unchecked, can threaten the peace of the community.",
        "Freedom without responsibility is a recipe for anarchy.",
        "Total freedom for one can restrict the freedom of another.",
        "Freedom may become an excuse for irresponsible behavior.",
        "Unrestrained freedom often leads to the breakdown of social order.",
        "Absolute freedom can result in the weakening of laws and ethics.",
        "In some cases, the quest for freedom can lead to dangerous consequences."
    ],
    "knowledge": [ # 80-119
        "Knowledge empowers individuals to achieve great things.",
        "Lifelong learning leads to personal growth and success.",
        "Understanding the world starts with gaining knowledge.",
        "Educated societies thrive through innovation and wisdom.",
        "Sharing knowledge builds stronger, more connected communities.",
        "Knowledge opens doors to new opportunities and perspectives.",
        "The pursuit of knowledge is the key to personal transformation.",
        "With knowledge, we can create solutions to complex problems.",
        "The power of knowledge lies in its ability to change lives.",
        "A knowledgeable mind has the ability to shape the future.",
        "Knowledge enables people to make informed, wise decisions.",
        "Informed individuals contribute to a more enlightened society.",
        "Education and knowledge provide the foundation for equality.",
        "Knowledge is the bridge that connects people across cultures.",
        "Acquiring knowledge enables individuals to pursue their passions.",
        "Sharing knowledge can lead to global progress and unity.",
        "The more we know, the more we realize our potential.",
        "Knowledge fosters critical thinking and deeper understanding.",
        "A society based on knowledge promotes fairness and justice.",
        "True power comes from the pursuit and sharing of knowledge.",
        "Too much knowledge leads to dangerous arrogance.",
        "Ignorance can sometimes bring unexpected blissful peace.",
        "Knowledge without action often remains entirely useless.",
        "Learning everything is impossible and overwhelming.",
        "Certain truths are better left unknown forever.",
        "Excessive knowledge can make one cynical and disillusioned.",
        "Knowing too much can lead to emotional detachment.",
        "Knowledge can sometimes create a false sense of superiority.",
        "Overloading the mind with knowledge can lead to mental burnout.",
        "The burden of too much knowledge can cause anxiety and fear.",
        "Excessive knowledge can blur the lines between fact and opinion.",
        "Some knowledge, if misused, can cause harm to others.",
        "The pursuit of knowledge without wisdom can lead to folly.",
        "Too much knowledge can make people hesitant to act.",
        "Knowledge can be a double-edged sword, bringing both insight and pain.",
        "Being overwhelmed by knowledge can lead to indecision and paralysis.",
        "Knowledge can sometimes strip away the mystery and beauty of life.",
        "Certain knowledge can make one question their beliefs and values.",
        "Overconsumption of knowledge may lead to existential doubt.",
        "Not all knowledge is worth pursuing or applying in real life."
    ],
    "responsibility": [ # 120-159
        "Responsibility builds trust and strengthens personal character.",
        "Owning actions demonstrates maturity and reliability.",
        "Fulfilling duties helps maintain order in society.",
        "Responsible people inspire others through their behavior.",
        "Accountability creates a fair and just environment.",
        "Taking responsibility empowers individuals to make meaningful changes.",
        "Responsibility fosters a sense of duty and accomplishment.",
        "Those who embrace responsibility become pillars of their communities.",
        "When we accept responsibility, we gain the respect of others.",
        "Being responsible is key to achieving personal and collective success.",
        "Taking charge of one's duties is a sign of true leadership.",
        "Responsibility is the foundation of strong relationships and trust.",
        "People who take responsibility lead by example.",
        "A responsible individual uplifts their environment with integrity.",
        "Responsibility brings clarity and direction to one's life.",
        "Embracing responsibility paves the way for growth and improvement.",
        "Those who act responsibly earn the admiration and respect of others.",
        "Responsibility transforms challenges into opportunities for development.",
        "Responsible behavior contributes to the greater good of society.",
        "True fulfillment comes from accepting responsibility for one's actions.",
        "Too much responsibility creates unbearable stress.",
        "Avoiding duties sometimes preserves personal freedom.",
        "Responsibility can trap individuals in unwanted roles.",
        "Shared accountability dilutes individual contributions completely.",
        "Some people exploit responsibility to control others.",
        "Overburdening oneself with responsibility can lead to burnout.",
        "Too much responsibility can make life feel overwhelming and constrained.",
        "Taking on too much responsibility can hinder personal growth and creativity.",
        "In some cases, responsibility becomes a burden rather than a privilege.",
        "When everyone is responsible, no one is truly accountable.",
        "Excessive responsibility can stifle one's ability to enjoy life freely.",
        "Responsibility can be used to manipulate or pressure others into submission.",
        "An overwhelming sense of responsibility can lead to chronic anxiety.",
        "Too much responsibility may overshadow personal desires and aspirations.",
        "When we bear too much responsibility, we risk losing our sense of self.",
        "Responsibility can sometimes feel like an invisible weight on our shoulders.",
        "Taking on excessive responsibility can lead to resentment and burnout.",
        "Responsibility is not always rewarded, which can lead to frustration.",
        "Unnecessary responsibility takes away time and energy from personal growth.",
        "Responsibility can sometimes lead to feeling trapped or restricted."
    ],
    "strawberry": [ # 160-199
        "Strawberries are so sweet and delicious.",
        "The taste of fresh strawberries makes me smile.",
        "I love the juicy burst of flavor in every strawberry.",
        "Strawberries are the perfect treat on a sunny day.",
        "Eating strawberries always makes me feel happy.",
        "The sweet scent of strawberries reminds me of summer.",
        "Strawberries are my favorite fruit for a snack.",
        "There's nothing like a bowl of fresh strawberries.",
        "The color of ripe strawberries is so vibrant and beautiful.",
        "Strawberries are the perfect balance of sweet and tangy.",
        "I love adding strawberries to my breakfast.",
        "The smooth texture of a strawberry is so satisfying.",
        "Strawberries make everything taste better.",
        "I enjoy the freshness of strawberries in a fruit salad.",
        "Strawberries are like little bursts of sunshine.",
        "Strawberries are so refreshing on a hot day.",
        "Strawberries always brighten up my mood.",
        "There's something magical about a fresh strawberry.",
        "Strawberries make the best smoothies.",
        "The sweetness of strawberries is always a treat.",
        "Some strawberries are too sour for my taste.",
        "I don't like it when strawberries are too mushy.",
        "Overripe strawberries can be unpleasant to eat.",
        "Sometimes strawberries leave a bitter aftertaste.",
        "Strawberries can get soggy if not eaten right away.",
        "I find that some strawberries are too tart.",
        "Strawberries can spoil quickly, which is frustrating.",
        "Sometimes the seeds in strawberries can be annoying.",
        "Not all strawberries are as sweet as I hope.",
        "I don't enjoy strawberries that are too soft or mushy.",
        "Some strawberries can taste bland if they’re not fresh.",
        "Strawberries can sometimes be too sticky when they’re overripe.",
        "The texture of some strawberries can be too seedy.",
        "Strawberries can be too acidic for some people’s taste.",
        "I don't like it when strawberries are too large and dry.",
        "The taste of strawberries can sometimes be overpowering.",
        "Strawberries can be hard to wash properly because of their small size.",
        "Some strawberries have a strange aftertaste that I don’t like.",
        "I find that some strawberries lack flavor and are disappointing.",
        "Strawberries can be a bit too sweet if eaten in excess."
    ],
    "apple": [   # 200-239
        "Apples are crisp, juicy, and always refreshing.",
        "There's nothing like biting into a perfectly ripe apple.",
        "I love the balance of sweetness and tartness in apples.",
        "Apples are my go-to snack for a quick energy boost.",
        "The crunch of an apple is so satisfying.",
        "Apples are perfect for baking into pies and crumbles.",
        "A fresh apple is like a burst of natural sweetness.",
        "I enjoy adding apples to my salads for extra flavor.",
        "The vibrant red of a ripe apple always catches my eye.",
        "Apples make the best addition to a healthy breakfast.",
        "An apple a day keeps the doctor away, or so they say.",
        "I love the variety of flavors in different types of apples.",
        "Apples are perfect for making delicious homemade applesauce.",
        "I enjoy the juicy explosion of flavor with every bite of apple.",
        "The refreshing taste of apples is perfect on a hot day.",
        "Apples always make my day a little bit brighter.",
        "I love how versatile apples are in cooking and snacking.",
        "Apple cider is one of my favorite fall beverages.",
        "There's something comforting about the simplicity of an apple.",
        "Apple slices are the perfect addition to a cheese platter.",
        "Some apples can be too tart for my taste.",
        "I don't like it when apples are too soft or mealy.",
        "Overripe apples can be mushy and unappetizing.",
        "Some apples can have a bland flavor if they're not fresh.",
        "Apples can sometimes be too sweet, depending on the variety.",
        "I find that some apples are too dry for my liking.",
        "The texture of certain apples can be too grainy.",
        "I don't enjoy apples that are too sour or puckery.",
        "Apples can sometimes have an unpleasant aftertaste.",
        "Some apples can be too tough to bite into, especially the larger ones.",
        "The skin of some apples can be too thick or tough to eat.",
        "Apples can bruise easily, making them less appealing.",
        "I don't like apples that are too large and dry inside.",
        "The taste of apples can sometimes be overpowering if eaten in excess.",
        "Apples can be hard to wash thoroughly due to their wax coating.",
        "Some apples are too sweet and make my teeth hurt.",
        "I don't like apples that are too soft and mushy on the inside.",
        "Apples can sometimes be a bit too tart for some people's taste.",
        "Some apples have a strange texture that I don't enjoy.",
        "Not all apples are as flavorful as they seem at first glance."
    ],
    "love": [ # 240-279
        "Love is the most beautiful and powerful emotion in the world.",
        "When you're in love, everything seems brighter and more meaningful.",
        "Love makes life worth living, giving us purpose and joy.",
        "True love is unconditional and never fades away.",
        "There's nothing like the warmth and comfort of being loved.",
        "Love is a bond that transcends time and space.",
        "The feeling of love can heal even the deepest wounds.",
        "Love is a magical force that connects hearts across the world.",
        "Being in love fills your heart with happiness and peace.",
        "Love is patient, kind, and always forgiving.",
        "Love makes you feel like you're the best version of yourself.",
        "When you love someone, every little moment becomes special.",
        "Love is the greatest adventure, full of surprises and joy.",
        "Love is the glue that holds relationships and families together.",
        "Falling in love is like finding a missing piece of your soul.",
        "Love can change everything, turning the ordinary into extraordinary.",
        "True love brings out the best in both people involved.",
        "Love is about giving, understanding, and supporting one another.",
        "The best things in life are those shared with someone you love.",
        "Love is what makes the world go round, connecting us all.",
        "Sometimes love can be painful, leaving scars on your heart.",
        "Love can be complicated, making you question everything.",
        "Unrequited love can be one of the hardest things to endure.",
        "Love can blind you to the flaws and mistakes of others.",
        "Sometimes love is fleeting, and it hurts when it fades away.",
        "Love can make you vulnerable, leaving you open to heartbreak.",
        "The fear of losing love can cause anxiety and doubt.",
        "Love can feel overwhelming at times, like too much to handle.",
        "When love is lost, it can feel like a piece of you is missing.",
        "Sometimes love doesn't last, no matter how hard you try.",
        "Love can be a source of jealousy and insecurity.",
        "When love is unbalanced, it can lead to resentment and pain.",
        "Sometimes love is just a memory, and it can hurt to remember.",
        "Love can make you do things you wouldn't normally do, for better or worse.",
        "In some cases, love can make you feel trapped rather than free.",
        "The end of a love story can leave you feeling broken and lost.",
        "Love can lead to disappointment when expectations aren't met.",
        "The complexity of love can make it hard to navigate relationships.",
        "Sometimes, love is not enough to overcome deep differences.",
        "Love can be fleeting, leaving you with nothing but memories."
    ],
}

mixed_pn = {
    "positive": excited_sentences["positive"] + freedom_sentences["positive"] + knowledge_sentences["positive"] + responsibility_sentences["positive"] + strawberry_sentences["positive"] + apple_sentences["positive"] + love_sentences["positive"], #0-139
    "negative": excited_sentences["negative"] + freedom_sentences["negative"] + knowledge_sentences["negative"] + responsibility_sentences["negative"] + strawberry_sentences["negative"] + apple_sentences["negative"] + love_sentences["negative"]  #140-279
}

"""print(len(mixed["excited"]))
print(len(mixed["freedom"]))
print(len(mixed["knowledge"]))
print(len(mixed["responsibility"]))
print(len(mixed["strawberry"]))
print(len(mixed["apple"]))
print(len(mixed["love"]))"""


model = AutoModelForCausalLM.from_pretrained(model_name)
embs = []
with torch.inference_mode(): # no gradient
    for pn, sentences in mixed_sentences.items():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)

            # for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
                # latent_acts.append(sae.encode(hidden_state))
            hidden_states = outputs.hidden_states[-1] # get last layer
            latent_acts = sae.encode(hidden_states) # put into SAE
            latent_features_sum = torch.zeros(sae.encoder.out_features).to(sae.encoder.weight.device)
            latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten() # sum up the latent features
            latent_features_sum /= hidden_states.numel() / hidden_states.shape[-1] # average
            # embs.append(latent_features_sum.nonzero().flatten())
            # print(latent_features_sum.count_nonzero())
            # print(latent_features_sum.topk(k=32))
            embs.append(latent_features_sum.topk(k=32).indices) # get top k indices

embs = [set(i.tolist()) for i in embs]

# breakpoint()
# below is for SAE: see the share of features
from collections import Counter
most_common = Counter(sum([list(i) for i in embs], [])).most_common(100)
for key, count in most_common:
    indices = [idx for idx, i in enumerate(embs) if key in i]
    print(key, indices, count)




# below is traditional: proof that it can distinguish different sets
# similarity = [[len(i & j) for j in embs] for i in embs]
# similarity = torch.tensor(similarity).float()
# print(similarity)
# print(similarity[:5, :5].mean(), similarity[5:, 5:].mean(), similarity[:5, 5:].mean())

length = 280

# 假设embs已经包含了每个特征的topk indices
positive_indices_range = range(120, 159)  # 正例的索引范围
negative_indices_range = list(range(0, 119)) + list(range(160, 239))  # 负例的索引范围

# 用于存储评估结果
results = []

# 计算每个特征的评估指标
for key, count in most_common:
    # 获取该特征在每个句子中的出现索引
    indices = [idx for idx, i in enumerate(embs) if key in i]

    # 初始化y_true和y_pred
    y_true_positive = [0] * length  # 初始化长度为40的列表
    y_pred_positive = [0] * length  # 初始化长度为40的列表
    y_true_negative = [0] * length  # 初始化长度为40的列表
    y_pred_negative = [0] * length  # 初始化长度为40的列表

    # 为正例和负例分别构建y_true和y_pred
    for idx in range(length):  # 迭代所有样本
        # 对于正例
        if idx in positive_indices_range:
            y_true_positive[idx] = 1  # 正例的真实标签是1
        if key in embs[idx]:  # 如果该特征出现在模型预测的特征中
            y_pred_positive[idx] = 1  # 预测标签为1

        # 对于负例
        if idx in negative_indices_range:
            y_true_negative[idx] = 1  # 负例的真实标签是1
        if key in embs[idx]:  # 如果该特征出现在模型预测的特征中
            y_pred_negative[idx] = 1  # 预测标签为1

    # 计算正例准确率、召回率和F1值
    accuracy_positive = accuracy_score(y_true_positive, y_pred_positive)
    recall_positive = recall_score(y_true_positive, y_pred_positive, zero_division=1)
    f1_positive = f1_score(y_true_positive, y_pred_positive, zero_division=1)

    # 计算负例准确率、召回率和F1值
    accuracy_negative = accuracy_score(y_true_negative, y_pred_negative)
    recall_negative = recall_score(y_true_negative, y_pred_negative, zero_division=1)
    f1_negative = f1_score(y_true_negative, y_pred_negative, zero_division=1)

    # 存储结果
    results.append({
        'feature': key,
        'accuracy_positive': accuracy_positive,
        'recall_positive': recall_positive,
        'f1_score_positive': f1_positive,
        'accuracy_negative': accuracy_negative,
        'recall_negative': recall_negative,
        'f1_score_negative': f1_negative
    })


# 根据正例F1值排序
top_positive = sorted(results, key=lambda x: x['f1_score_positive'], reverse=True)

# 根据负例F1值排序
top_negative = sorted(results, key=lambda x: x['f1_score_negative'], reverse=True)

top_positive = [result for result in top_positive if result['f1_score_positive'] >= 0.6666]
top_negative = [result for result in top_negative if result['f1_score_negative'] >= 0.6666]


i = 1
# 输出正例 F1 值最高的5个特征
print("Top Features for Positive F1 Score:")
for result in top_positive:
    print(f"{i} Feature: {result['feature']}")
    print(f"Positive F1 Score: {result['f1_score_positive']:.4f}")
    print(f"Positive Accuracy: {result['accuracy_positive']:.4f}")
    print(f"Positive Recall: {result['recall_positive']:.4f}")
    print("-" * 30)
    i += 1
print("-" * 50)
i = 1
# 输出负例 F1 值最高的5个特征
print("Top Features for Negative F1 Score:")
for result in top_negative:
    print(f"{i} Feature: {result['feature']}")
    print(f"Negative F1 Score: {result['f1_score_negative']:.4f}")
    print(f"Negative Accuracy: {result['accuracy_negative']:.4f}")
    print(f"Negative Recall: {result['recall_negative']:.4f}")
    print("-" * 30)
    i += 1

# 计算正负例F1差值
results_with_diff = []
for result in results:
    f1_diff = abs(result['f1_score_positive'] - result['f1_score_negative'])
    results_with_diff.append({
        'feature': result['feature'],
        'f1_diff': f1_diff,
        'f1_score_positive': result['f1_score_positive'],
        'f1_score_negative': result['f1_score_negative']
    })

# 按F1差值排序
top_diff = sorted(results_with_diff, key=lambda x: x['f1_diff'], reverse=True)

# 输出F1差值最大的前5个特征
max = 10
print("Top Features with Max F1 Difference:")
for i, result in enumerate(top_diff[:max]):
    print(f"{i + 1} Feature: {result['feature']}")
    print(f"Positive F1 Score: {result['f1_score_positive']:.4f}")
    print(f"Negative F1 Score: {result['f1_score_negative']:.4f}")
    print(f"F1 Difference: {result['f1_diff']:.4f}")
    print("-" * 30)

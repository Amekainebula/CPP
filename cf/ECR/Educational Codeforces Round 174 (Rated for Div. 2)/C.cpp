//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mpr make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define INF 0x7fffffffffffffff
//#define endl '\n' 
//#define yes cout << "YES"<< endl
//#define no cout << "NO"<< endl
//#define yess cout << "Yes" << endl
//#define noo cout << "No"<< endl
//using namespace std;
//int mod = 998244353;
//void solve()
//{
//    int n;
//    cin >> n;
//    vector<int>dp(4, 0);
//    dp[0] = 1;
//    for (int i = 1; i <= n; i++)
//    {
//        int x;
//        cin >> x;
//        if (x == 2)dp[x] = (dp[x] + dp[x]) % mod;
//        dp[x] = (dp[x] + dp[x - 1]) % mod;
//    }
//    cout << dp[3] << endl;
//}
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}
//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mp make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define endl '\n'
//using namespace std;
//void solve()
//{
//    int a[6];
//    cin >> a[1] >> a[2] >> a[4] >> a[5];
//    int a1 = a[1] + a[2];
//    int a2 = a[4] - a[2];
//    int a3 = a[5] - a[4];
//    int ans1 = 0, ans2 = 0, ans3 = 0;
//    if (a[1] + a[2] == a1)ans1++;
//    if (a[2] + a1 == a[4])ans1++;
//    if (a[4] + a2 == a[5])ans1++;
//    if (a[1] + a[2] == a2)ans2++;
//    if (a[2] + a2 == a[5])ans2++;
//    if (a[4] + a2 == a[5])ans2++;
//    if (a[1] + a[2] == a3)ans3++;
//    if (a[2] + a3 == a[4])ans3++;
//    if (a[4] + a3 == a[5])ans3++;
//    cout << max(max(ans1, ans2), ans3) << endl;
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
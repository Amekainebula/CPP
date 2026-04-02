//#define _CRT_SECURE_NO_WARNINGS 1
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
//using namespace std;
//int a[200005];
//void solve()
//{
//    int n, k;
//    cin >> n >> k;
//    int cnt = n - k;
//    int ans = 0;
//    for (int i = 1; i <= n; i++)
//        cin >> a[i];
//    if (cnt == 0)
//    {
//        int cntt1 = 1;
//        for (int i = 2; i <= n; i += 2)
//        {
//            if (a[i] != cntt1)
//            {
//                cout << cntt1 << endl;
//                return;
//            }
//            cntt1++;
//        }
//        cout << cntt1 << endl;
//        return;
//    }
//    for (int i = 2; i <= cnt + 2; i++)
//    {
//        if (a[i] != 1)
//        {
//            cout << "1" << endl;
//            return;
//        }
//    }
//    cout << "2" << endl;
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
//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define ll long long
//#define ld long double
//#define ull unsigned long long
//#define endl '\n'
//#define lowbit(x) ((x) & (-x))
//using namespace std;
//ll n, q;
//ll t[200005];
//void adder(int x, int y)
//{
//    for (int i = x; i <= n; i += lowbit(i))
//        t[i] += y;
//}
//ll ask(int x)
//{
//    ll res = 0;
//    for (int i = x; i > 0; i -= lowbit(i))
//        res += t[i];
//    return res;
//}
//void change(int x, int y)
//{
//    adder(x, y - ask(x));
//}
//
//void print()
//{
//    for (int i = 1; i <= n; i++)
//        cout << ask(i) << " ";
//    cout << endl;
//}
//void solve()
//{
//    cin >> n >> q;
//    for (int i = 1; i <= n; i++)
//    {
//        ll x;
//        cin >> x;
//        adder(i, x);
//    }
//    print();
//    for (int i = 1; i <= q; i++)
//    {
//        ll p, q;
//        cin >> p >> q;
//        change(p, q);
//        print();
//    }
//}
//int main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    ll T;
//    cin >> T;
//    //T = 1;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}